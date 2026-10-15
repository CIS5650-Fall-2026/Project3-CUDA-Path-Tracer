#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

class Pipeline
{
public:
    // Every shader stage that will read push constants
    static constexpr VkShaderStageFlags kPushConstantStages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR
        | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    Pipeline() = default;
    ~Pipeline()
    {
        destroy();
    }

    // Pipeline should be created once and passed by reference as needed,
    // so we should disable copies/moves
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

    bool init(VulkanContext& context);
    void destroy();

    VkDescriptorSetLayout getRendererDescriptorSetLayout() const
    {
        return m_rendererDescriptorSetLayout;
    }

    VkPipelineLayout getPipelineLayout() const
    {
        return m_pipelineLayout;
    }

    VkPipeline getHandle() const
    {
        return m_pipeline;
    }

    const VkStridedDeviceAddressRegionKHR& getRaygenDeviceAddrRegion() const
    {
        return m_raygenDeviceAddrRegion;
    }

    const VkStridedDeviceAddressRegionKHR& getMissDeviceAddrRegion() const
    {
        return m_missDeviceAddrRegion;
    }

    const VkStridedDeviceAddressRegionKHR& getHitDeviceAddrRegion() const
    {
        return m_hitDeviceAddrRegion;
    }

    const VkStridedDeviceAddressRegionKHR& getCallableDeviceAddrRegion() const
    {
        return m_callableDeviceAddrRegion;
    }

private:
    // Absolute SPIR-V output directory, set by CMake on-build
    static constexpr const char* kShaderDirectoryPath = SHADER_DIR;

    // Shader group indices. Groups are created in this order, and the SBT copies
    // each group's handle into the matching region.
    static constexpr uint32_t kRaygenGroupIdx = 0;
    static constexpr uint32_t kMissGroupIdx = 1;
    static constexpr uint32_t kClosestHitGroupIdx = 2;
    static constexpr uint32_t kShaderGroupCount = 3;

    VulkanContext* m_context = nullptr;

    VkDescriptorSetLayout m_rendererDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkBuffer m_sbtBuffer = VK_NULL_HANDLE;
    VmaAllocation m_sbtAllocation = VK_NULL_HANDLE;

    // SBT Device Address Regions
    VkStridedDeviceAddressRegionKHR m_raygenDeviceAddrRegion{};
    VkStridedDeviceAddressRegionKHR m_missDeviceAddrRegion{};
    VkStridedDeviceAddressRegionKHR m_hitDeviceAddrRegion{};
    VkStridedDeviceAddressRegionKHR m_callableDeviceAddrRegion{};

    bool createRendererDescriptorSetLayout();
    bool createPipelineLayout();
    bool createPipeline();
    bool createShaderBindingTable();

    static bool createVkShaderModule(VkDevice device, const std::vector<uint32_t>& code, VkShaderModule& shaderModule);
    static bool readSpirvFile(const std::string& filename, std::vector<uint32_t>& code);
};