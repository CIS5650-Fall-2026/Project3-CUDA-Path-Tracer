#include "pipeline.h"

#include "context.h"
#include "gpu/shared.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

bool Pipeline::init(VulkanContext& context)
{
    m_context = &context;

    if (!createRendererDescriptorSetLayout())
    {
        destroy();
        return false;
    }

    if (!createPipelineLayout())
    {
        destroy();
        return false;
    }

    if (!createPipeline())
    {
        destroy();
        return false;
    }

    if (!createShaderBindingTable())
    {
        destroy();
        return false;
    }

    return true;
}

void Pipeline::destroy()
{
    if (m_context == nullptr)
    {
        return;
    }

    // Destroy in reverse order of creation
    vmaDestroyBuffer(m_context->getAllocator(), m_sbtBuffer, m_sbtAllocation);
    m_sbtBuffer = VK_NULL_HANDLE;
    m_sbtAllocation = VK_NULL_HANDLE;

    VkDevice device = m_context->getDevice();
    vkDestroyPipeline(device, m_pipeline, nullptr);
    m_pipeline = VK_NULL_HANDLE;

    vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;

    vkDestroyDescriptorSetLayout(device, m_rendererDescriptorSetLayout, nullptr);
    m_rendererDescriptorSetLayout = VK_NULL_HANDLE;

    m_raygenDeviceAddrRegion = {};
    m_missDeviceAddrRegion = {};
    m_hitDeviceAddrRegion = {};
    m_callableDeviceAddrRegion = {};
    m_context = nullptr;
}

bool Pipeline::createRendererDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    // Configure a binding for our TLAS
    bindings.at(0).binding = gpu::kTLASBinding;
    bindings.at(0).descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings.at(0).descriptorCount = 1;
    bindings.at(0).stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings.at(0).pImmutableSamplers = nullptr;

    // Configure a binding for our accumulation image (holds a running sum of samples)
    bindings.at(1).binding = gpu::kAccumImageBinding;
    bindings.at(1).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.at(1).descriptorCount = 1;
    bindings.at(1).stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings.at(1).pImmutableSamplers = nullptr;

    // Configure a binding for our display image (holds the render output that gets
    // blitted into a swapchain image prior to presentation)
    bindings.at(2).binding = gpu::kDisplayImageBinding;
    bindings.at(2).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.at(2).descriptorCount = 1;
    bindings.at(2).stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings.at(2).pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    createInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_context->getDevice(), &createInfo, nullptr, &m_rendererDescriptorSetLayout)
        != VK_SUCCESS)
    {
        m_rendererDescriptorSetLayout = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create renderer descriptor set layout.\n");
        return false;
    }

    return true;
}

bool Pipeline::createPipelineLayout()
{
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = kPushConstantStages;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(gpu::PushConstants);

    VkPipelineLayoutCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    
    // A descriptor set's number is its index in pSetLayouts, so enforce
    // that contract here
    std::array<VkDescriptorSetLayout, 1> setLayouts{};
    setLayouts.at(gpu::kRendererDescriptorSet) = m_rendererDescriptorSetLayout;

    createInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    createInfo.pSetLayouts = setLayouts.data();

    createInfo.pushConstantRangeCount = 1;
    createInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(m_context->getDevice(), &createInfo, nullptr, &m_pipelineLayout)
        != VK_SUCCESS)
    {
        m_pipelineLayout = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create pipeline layout.\n");
        return false;
    }

    return true;
}

bool Pipeline::createPipeline()
{
    const uint32_t raygenStageIdx = 0;
    const uint32_t missStageIdx = 1;
    const uint32_t closestHitStageIdx = 2;
    const uint32_t shaderStageCount = 3;

    const std::array<VkShaderStageFlagBits, shaderStageCount> shaderStageFlags = {
        VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        VK_SHADER_STAGE_MISS_BIT_KHR,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
    };

    const std::array<const char*, shaderStageCount> shaderStageFilenames = {
        "raygen.rgen.spv",
        "miss.rmiss.spv",
        "mesh.rchit.spv"
    };

    std::array<VkShaderModule, shaderStageCount> shaderModules{};
    std::array<VkPipelineShaderStageCreateInfo, shaderStageCount> shaderStages{};
    VkDevice device = m_context->getDevice();

    bool creationSuccessful = true;
    for (uint32_t i = 0; i < shaderStageCount; i++)
    {
        const std::string path = std::string(kShaderDirectoryPath) + "/" + shaderStageFilenames.at(i);

        std::vector<uint32_t> code;
        creationSuccessful = readSpirvFile(path, code)
            && createVkShaderModule(device, code, shaderModules.at(i));

        if (!creationSuccessful)
        {
            break;
        }

        shaderStages.at(i).sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages.at(i).stage = shaderStageFlags.at(i);
        shaderStages.at(i).module = shaderModules.at(i);
        shaderStages.at(i).pName = "main";
    }

    if (creationSuccessful)
    {
        // We need to create one shader group per SBT record
        std::array<VkRayTracingShaderGroupCreateInfoKHR, kShaderGroupCount> groupCreateInfos{};

        // Explicitly label each shader as unused--we'll set them case-by-case afterwards
        for (VkRayTracingShaderGroupCreateInfoKHR& groupCreateInfo : groupCreateInfos)
        {
            groupCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            groupCreateInfo.generalShader = VK_SHADER_UNUSED_KHR;
            groupCreateInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
            groupCreateInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
            groupCreateInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
        }

        // The raygen and miss shaders fall under the "general" group type, which indicates that
        // the group contains exactly one shader that runs on its own
        groupCreateInfos.at(kRaygenGroupIdx).type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groupCreateInfos.at(kRaygenGroupIdx).generalShader = raygenStageIdx;

        groupCreateInfos.at(kMissGroupIdx).type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groupCreateInfos.at(kMissGroupIdx).generalShader = missStageIdx;

        // The closest-hit shader falls under the "triangles hit" group type since we're using
        // triangle geometry
        groupCreateInfos.at(kClosestHitGroupIdx).type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groupCreateInfos.at(kClosestHitGroupIdx).closestHitShader = closestHitStageIdx;

        VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.groupCount = static_cast<uint32_t>(groupCreateInfos.size());
        pipelineCreateInfo.pGroups = groupCreateInfos.data();

        // Raygen will exclusively handle ray bounces/iteration
        pipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
        pipelineCreateInfo.layout = m_pipelineLayout;

        if (vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &m_pipeline)
            != VK_SUCCESS)
        {
            m_pipeline = VK_NULL_HANDLE;
            fprintf(stderr, "Failed to create ray tracing pipeline.\n");
            creationSuccessful = false;
        }
    }

    // Shader modules aren't needed after pipeline creation, so delete them here
    for (VkShaderModule shaderModule : shaderModules)
    {
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    return creationSuccessful;
}

bool Pipeline::createShaderBindingTable()
{
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProperties = m_context->getRtProperties();

    const uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    const uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;
    const uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;

    // Define a lambda that rounds a value up to the next multiple of alignment.
    // Note that this only works because Vulkan guarantees alignment to be
    // a power of two.
    auto alignUp = [](VkDeviceSize value, VkDeviceSize alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    };

    std::vector<uint8_t> handles(kShaderGroupCount * handleSize);
    if (vkGetRayTracingShaderGroupHandlesKHR(device, m_pipeline, 0, kShaderGroupCount, handles.size(), handles.data())
        != VK_SUCCESS)
    {
        fprintf(stderr, "Failed to get shader group handles.\n");
        return false;
    }

    // Each region holds one or more records, and each record contains only a group
    // handle, so the stride is the handle size rounded up to a multiple of handleAlignment.
    // Each region must also start at a multiple of baseAlignment.
    const VkDeviceSize recordStride = alignUp(handleSize, handleAlignment);

    // For each region we compute its offset (in bytes) within the SBT buffer we are going to create
    // NCHORTEK TODO: offset computation may need to be updated if additional shader records
    // are added to our shader groups (adding a single recordStride assumes each region only
    // contains a single record).
    const VkDeviceSize raygenOffset = 0;
    const VkDeviceSize missOffset = alignUp(raygenOffset + recordStride, baseAlignment);
    const VkDeviceSize hitOffset = alignUp(missOffset + recordStride, baseAlignment);
    const VkDeviceSize bufferSize = hitOffset + recordStride;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Allocate host-writable memory with persistently mapped, so the handles can
    // be written directly
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    // Every SBT region's address must be a multiple of baseAlignment, but a buffer is
    // only guaranteed its own (possibly smaller) memory alignment. Each region's
    // address is the buffer's base address plus an offset that's a multiple of
    // baseAlignment (raygen's is 0), so aligning the buffer's allocation to
    // baseAlignment makes every region address a multiple of baseAlignment too.
    VmaAllocationInfo allocInfo{};
    if (vmaCreateBufferWithAlignment(
        allocator,
        &bufferCreateInfo,
        &allocCreateInfo,
        baseAlignment,
        &m_sbtBuffer,
        &m_sbtAllocation,
        &allocInfo) != VK_SUCCESS)
    {
        m_sbtBuffer = VK_NULL_HANDLE;
        m_sbtAllocation = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create shader binding table buffer.\n");
        return false;
    }

    // Copy each group's handle into the record at the start of its region
    uint8_t* sbtData = static_cast<uint8_t*>(allocInfo.pMappedData);
    memcpy(sbtData + raygenOffset, handles.data() + kRaygenGroupIdx * handleSize, handleSize);
    memcpy(sbtData + missOffset, handles.data() + kMissGroupIdx * handleSize, handleSize);
    memcpy(sbtData + hitOffset, handles.data() + kClosestHitGroupIdx * handleSize, handleSize);

    // Make the writes visible to the device in case the memory isn't host-coherent
    // (host-coherent memory is already GPU-visible, so it'd be a no-op in that case)
    if (vmaFlushAllocation(allocator, m_sbtAllocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS)
    {
        fprintf(stderr, "Failed to flush shader binding table memory.\n");
        return false;
    }

    VkBufferDeviceAddressInfo addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = m_sbtBuffer;
    const VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(device, &addressInfo);

    // VkStridedDeviceAddressRegionKHR = { deviceAddress, stride, size }
    // The raygen region's size must equal its stride, because there can only ever be
    // a single raygen shader.
    m_raygenDeviceAddrRegion = { sbtAddress + raygenOffset, recordStride, recordStride };
    m_missDeviceAddrRegion = { sbtAddress + missOffset, recordStride, recordStride };
    m_hitDeviceAddrRegion = { sbtAddress + hitOffset, recordStride, recordStride };

    // We dont have any callable shaders, so zero-out its device address region to
    // indicate it is unused
    m_callableDeviceAddrRegion = {};

    return true;
}

bool Pipeline::readSpirvFile(const std::string& filename, std::vector<uint32_t>& code)
{
    // std::ios::ate instructs ifstream to start at the end of the file,
    // and std::ios::binary specifies the file to be read as its underlying
    // binary data
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        fprintf(stderr, "Failed to open shader file: %s\n", filename.c_str());
        return false;
    }

    // Because we start at the end of the file, we can quickly obtain
    // its size by obtaining the current position of the stream
    const std::streamoff fileSize = file.tellg();

    // Valid SPIR-V is a stream of 32-bit words, so sanity check that the
    // fileSize matches that expectation.
    if (fileSize <= 0 || static_cast<size_t>(fileSize) % sizeof(uint32_t) != 0)
    {
        fprintf(stderr, "Invalid SPIR-V file size (%lld bytes): %s\n",
            static_cast<long long>(fileSize),
            filename.c_str());
        return false;
    }

    // Ensure our code vector is large enough to store the full SPIR-V data
    code.resize(static_cast<size_t>(fileSize) / sizeof(uint32_t));

    // Seek back to the beginning of the file
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    file.close();
    return true;
}

bool Pipeline::createVkShaderModule(VkDevice device, const std::vector<uint32_t>& code, VkShaderModule& shaderModule)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    // codeSize needs to be in bytes
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();

    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        shaderModule = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create shader module.\n");
        return false;
    }

    return true;
}