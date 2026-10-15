#pragma once

#include <volk.h>

#include "vk/utils/buffer.h"
#include "sceneUtils.h"

#include <cstdint>
#include <vector>

class VulkanContext;

class GpuScene
{
public:
    GpuScene() = default;
    ~GpuScene()
    {
        destroy();
    }

    // GpuScene should be created once and passed by reference as needed,
    // so we should disable copies/moves
    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;
    GpuScene(GpuScene&&) = delete;
    GpuScene& operator=(GpuScene&&) = delete;

    bool init(VulkanContext& context, const sceneutil::FlatScene& flatScene);
    void destroy();

    uint64_t getSceneAddresses() const
    {
        return m_sceneAddressesBuffer.deviceAddress;
    }

    VkAccelerationStructureKHR getTLAS() const
    {
        return m_TLAS.handle;
    }

private:
    VulkanContext* m_context = nullptr;

    vkutil::Buffer m_vertexBuffer{};
    vkutil::Buffer m_indexBuffer{};
    vkutil::Buffer m_geometryBuffer{};
    vkutil::Buffer m_materialBuffer{};
    vkutil::Buffer m_lightBuffer{};
    vkutil::Buffer m_sceneAddressesBuffer{};

    struct AccelStructure
    {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        vkutil::Buffer buffer{};

        // The buffer device address of the acceleration structure,
        // which, unlike vkutil::Buffer.deviceAddress, must be queried
        // via vkGetAccelerationStructureDeviceAddressKHR
        VkDeviceAddress deviceAddress = 0;
    };

    // One TLAS for the scene, one BLAS per mesh
    std::vector<AccelStructure> m_BLASes;
    AccelStructure m_TLAS{};

    bool createBLASes(const sceneutil::FlatScene& flatScene);
    bool createTLAS(const sceneutil::FlatScene& flatScene);

    bool buildAccelStructure(
        VkAccelerationStructureTypeKHR type,
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& buildRanges,
        AccelStructure& accelStructure);

    void destroyAccelStructure(AccelStructure& accelStructure);

    bool createSceneDataBuffers(const sceneutil::FlatScene& flatScene);
    bool createSceneAddressesBuffer(uint32_t lightCount, float totalLightArea);
};