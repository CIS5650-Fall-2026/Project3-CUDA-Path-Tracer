#include "gpuscene.h"

#include "context.h"

#include <glm/gtc/type_ptr.hpp>

bool GpuScene::init(VulkanContext& context, const sceneutil::FlatScene& flatScene)
{
    m_context = &context;

    if (!createSceneDataBuffers(flatScene))
    {
        destroy();
        return false;
    }

    if (!createSceneAddressesBuffer(static_cast<uint32_t>(flatScene.lights.size()), flatScene.totalLightArea))
    {
        destroy();
        return false;
    }

    if (!createBLASes(flatScene))
    {
        destroy();
        return false;
    }

    if (!createTLAS(flatScene))
    {
        destroy();
        return false;
    }

    return true;
}

void GpuScene::destroy()
{
    if (m_context == nullptr)
    {
        return;
    }

    destroyAccelStructure(m_TLAS);

    for (AccelStructure& blas : m_BLASes)
    {
        destroyAccelStructure(blas);
    }

    m_BLASes.clear();

    vkutil::destroyBuffer(*m_context, m_vertexBuffer);
    vkutil::destroyBuffer(*m_context, m_indexBuffer);
    vkutil::destroyBuffer(*m_context, m_geometryBuffer);
    vkutil::destroyBuffer(*m_context, m_materialBuffer);
    vkutil::destroyBuffer(*m_context, m_lightBuffer);
    vkutil::destroyBuffer(*m_context, m_sceneAddressesBuffer);

    m_context = nullptr;
}

bool GpuScene::createSceneDataBuffers(const sceneutil::FlatScene& flatScene)
{
    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.vertices.data(),
        flatScene.vertices.size() * sizeof(gpu::Vertex),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_vertexBuffer))
    {
        fprintf(stderr, "Failed to create the vertex buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.indices.data(),
        flatScene.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_indexBuffer))
    {
        fprintf(stderr, "Failed to create the index buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.geometries.data(),
        flatScene.geometries.size() * sizeof(gpu::GeometryInfo),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_geometryBuffer))
    {
        fprintf(stderr, "Failed to create the geometry buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.materials.data(),
        flatScene.materials.size() * sizeof(gpu::GpuMaterial),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_materialBuffer))
    {
        fprintf(stderr, "Failed to create the material buffer on the GPU.\n");
        return false;
    }

    if (!flatScene.lights.empty()
        && !vkutil::createAndUploadBuffer(
            *m_context,
            flatScene.lights.data(),
            flatScene.lights.size() * sizeof(gpu::EmissiveTri),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            m_lightBuffer))
    {
        fprintf(stderr, "Failed to create the light buffer on the GPU.\n");
        return false;
    }

    return true;
}

bool GpuScene::createSceneAddressesBuffer(uint32_t lightCount, float totalLightArea)
{
    gpu::SceneAddresses sceneAddresses{};
    sceneAddresses.verticesAddr = m_vertexBuffer.deviceAddress;
    sceneAddresses.indicesAddr = m_indexBuffer.deviceAddress;
    sceneAddresses.geometriesAddr = m_geometryBuffer.deviceAddress;
    sceneAddresses.materialsAddr = m_materialBuffer.deviceAddress;
    sceneAddresses.lightsAddr = m_lightBuffer.deviceAddress;
    sceneAddresses.lightCount = lightCount;
    sceneAddresses.totalLightArea = totalLightArea;

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        &sceneAddresses,
        sizeof(sceneAddresses),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_sceneAddressesBuffer))
    {
        fprintf(stderr, "Failed to create the scene addresses buffer on the GPU.\n");
        return false;
    }

    return true;
}

bool GpuScene::createBLASes(const sceneutil::FlatScene& flatScene)
{
    m_BLASes.resize(flatScene.meshRanges.size());

    for (int meshIndex = 0; meshIndex < flatScene.meshRanges.size(); meshIndex++)
    {
        const std::vector<sceneutil::MeshPrimitiveRange>& primitiveRanges = flatScene.meshRanges.at(meshIndex);

        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        geometries.reserve(primitiveRanges.size());

        std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
        buildRanges.reserve(primitiveRanges.size());

        for (const sceneutil::MeshPrimitiveRange& primitiveRange : primitiveRanges)
        {
            // Describe our vertex data layout so the driver knows how to access the vertex
            // positions specifically
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = m_vertexBuffer.deviceAddress;
            triangles.vertexStride = sizeof(gpu::Vertex);

            // maxVertex is the highest index the driver will read
            triangles.maxVertex = primitiveRange.vertexCount - 1;

            triangles.indexType = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress = m_indexBuffer.deviceAddress;

            // BLAS doesn't require geometry transforms (placement is a TLAS job)
            triangles.transformData = {};

            VkAccelerationStructureGeometryKHR geometry{};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometry.geometry.triangles = triangles;

            // Opaque here means traversal never invokes an any-hit shader, which is
            // okay because we're only using a closest-hit shader for now
            geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

            VkAccelerationStructureBuildRangeInfoKHR buildRange{};
            buildRange.primitiveCount = primitiveRange.indexCount / 3;

            // primitiveOffset is in bytes
            buildRange.primitiveOffset = primitiveRange.indexOffset * sizeof(uint32_t);

            // firstVertex is an element count
            buildRange.firstVertex = primitiveRange.vertexOffset;
            buildRange.transformOffset = 0;

            geometries.push_back(geometry);
            buildRanges.push_back(buildRange);
        }

        if (!buildAccelStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometries,
            buildRanges,
            m_BLASes.at(meshIndex)))
        {
            fprintf(stderr, "Failed to build BLAS for mesh index %d.\n", meshIndex);
            return false;
        }
    }

    return true;
}

bool GpuScene::createTLAS(const sceneutil::FlatScene& flatScene)
{
    // VkTransformMatrixKHR is row-major 3x4, but glm::mat4 is column-major.
    // Transposing and copying sizeof(VkTransformMatrixKHR) drops the bottom row,
    // which is what VkTransformMatrixKHR expects
    auto toTransformMatrixKHR = [](const glm::mat4& matrix)
    {
        VkTransformMatrixKHR transform{};
        const glm::mat4 transposed = glm::transpose(matrix);
        memcpy(&transform, glm::value_ptr(transposed), sizeof(transform));
        return transform;
    };

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(flatScene.instanceFirstGeometry.size());

    for (int i = 0; i < flatScene.instances.size(); i++)
    {
        const MeshInstance& meshInstance = flatScene.instances.at(i);

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = toTransformMatrixKHR(meshInstance.modelMatrix);

        // instanceCustomIndex holds the instance's first row in geometries[]
        // The shader will read geometries[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT],
        // so instances sharing a BLAS can use different materials
        instance.instanceCustomIndex = flatScene.instanceFirstGeometry.at(i);

        instance.mask = gpu::kInstanceMaskAllBitsSet;

        // We only have one hit group for all geometry, so no offset necessary here
        instance.instanceShaderBindingTableRecordOffset = 0;

        // Backfaces must hit (both sides of a surface should be treated as valid intersections)
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        instance.accelerationStructureReference = m_BLASes.at(meshInstance.meshIndex).deviceAddress;

        instances.push_back(instance);
    }

    vkutil::Buffer instanceBuffer{};
    if (!vkutil::createAndUploadBuffer(
        *m_context,
        instances.data(),
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        instanceBuffer))
    {
        fprintf(stderr, "Failed to create the TLAS instance buffer.\n");
        return false;
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildRangeInfoKHR buildRange{};

    // primitiveCount counts whatever the geometry type's unit is.
    // For VK_GEOMETRY_TYPE_TRIANGLES_KHR its triangles, but for
    // VK_GEOMETRY_TYPE_INSTANCES_KHR its VkAccelerationStructureInstanceKHR
    // entries (one per MeshInstance)
    buildRange.primitiveCount = static_cast<uint32_t>(instances.size());

    // NCHORTEK NOTE: The BLAS builds each ran through endSingleTimeCommands,
    // which waits idle, so they are complete before this build reads them.
    // If the wait idle is every removed, this will need sync handling
    const bool buildSuccessful = buildAccelStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        { geometry },
        { buildRange },
        m_TLAS);

    // Right now the instanceBuffer is only needed to build the TLAS.
    // Because we never update the TLAS, we can safely destroy it
    // after the TLAS is created
    vkutil::destroyBuffer(*m_context, instanceBuffer);

    if (!buildSuccessful)
    {
        fprintf(stderr, "Failed to build the TLAS.\n");
        return false;
    }

    return true;
}

bool GpuScene::buildAccelStructure(
    VkAccelerationStructureTypeKHR accelStructType,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& buildRanges,
    AccelStructure& accelStruct)
{
    VkDevice device = m_context->getDevice();

    // We have one build range per geometry, and we use them
    // here to grab the maximum primitive count for the acceleration
    // structure we're creating. For a TLAS "primitive" here is a
    // mesh/BLAS instance. For a BLAS "primitive" is a triangle.
    std::vector<uint32_t> maxPrimitiveCounts;
    maxPrimitiveCounts.reserve(buildRanges.size());
    for (const VkAccelerationStructureBuildRangeInfoKHR& range : buildRanges)
    {
        maxPrimitiveCounts.push_back(range.primitiveCount);
    }

    // Filled out most of our build info now. We'll need the scratch buffer and 
    // destination accel struct to fill the rest
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = accelStructType;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount = static_cast<uint32_t>(geometries.size());
    buildInfo.pGeometries = geometries.data();

    // Query the size of the acceleration structure and scratch buffer
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        maxPrimitiveCounts.data(),
        &sizeInfo);

    // The driver builds its BVH into this buffer, so we dont have to
    // do it manually.
    if (!vkutil::createBuffer(
        *m_context,
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        0,
        accelStruct.buffer))
    {
        fprintf(stderr, "Failed to create acceleration structure storage buffer.\n");
        return false;
    }

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = accelStruct.buffer.bufferHandle;
    createInfo.offset = 0;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = accelStructType;

    if (vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &accelStruct.handle)
        != VK_SUCCESS)
    {
        accelStruct.handle = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create acceleration structure.\n");
        return false;
    }

    // Create the scratch buffer with the required driver-defined alignment
    vkutil::Buffer scratchBuffer{};
    if (!vkutil::createBuffer(
        *m_context,
        sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_context->getAccelProperties().minAccelerationStructureScratchOffsetAlignment,
        scratchBuffer))
    {
        fprintf(stderr, "Failed to create acceleration structure scratch buffer.\n");
        return false;
    }

    buildInfo.dstAccelerationStructure = accelStruct.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    // Build our acceleration structure with a one-off command buffer
    VkCommandBuffer commandBuffer = m_context->beginSingleTimeCommands();

    const VkAccelerationStructureBuildRangeInfoKHR* buildRangesPtr = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, &buildRangesPtr);

    m_context->endSingleTimeCommands(commandBuffer);

    // NCHORTEK TODO: this is only safe because endSingleTimeCommands currently performs
    // a queue-level waitIdle
    vkutil::destroyBuffer(*m_context, scratchBuffer);

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = accelStruct.handle;
    accelStruct.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

    return true;
}

void GpuScene::destroyAccelStructure(AccelStructure& accelStructure)
{
    vkDestroyAccelerationStructureKHR(m_context->getDevice(), accelStructure.handle, nullptr);
    vkutil::destroyBuffer(*m_context, accelStructure.buffer);
    accelStructure = {};
}