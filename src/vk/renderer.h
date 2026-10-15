#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include "gui.h"
#include "pipeline.h"
#include "gpu/shared.h"

#include <array>
#include <cstdint>

struct GLFWwindow;
class VulkanContext;
class Swapchain;
class GpuScene;

class Renderer
{
public:
    Renderer() = default;
    ~Renderer()
    {
        destroy();
    }

    // Renderer should be created once and passed by reference as needed,
    // so we should disable copies/moves
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    bool init(VulkanContext& context, Swapchain& swapchain, GLFWwindow* window, uint32_t width, uint32_t height, GpuScene& gpuScene);
    void destroy();
    bool drawFrame();

    Gui& getGui()
    {
        return m_gui;
    }

    void setCameraParams(const gpu::CameraParams& camera)
    {
        m_pushConstants.camera = camera;
    }

    void setDepthOfField(bool enabled)
    {
        if (enabled)
        {
            m_pushConstants.flags |= gpu::kFlagDepthOfField;
        }
        else
        {
            m_pushConstants.flags &= ~gpu::kFlagDepthOfField;
        }
    }

    void setMaxTraceDepth(uint32_t maxDepth)
    {
        m_pushConstants.maxDepth = maxDepth;
    }

    uint32_t getRenderedFrameCount() const
    {
        return m_pushConstants.renderedFrameCount;
    }

    void resetRenderedFrameCount()
    {
        m_pushConstants.renderedFrameCount = 0;
    }

private:
    static constexpr uint32_t kFramesInFlight = 2;
    static constexpr uint64_t kMaxTimeout = UINT64_MAX;

    VulkanContext* m_context = nullptr;
    Swapchain* m_swapchain = nullptr;
    Pipeline m_pipeline;
    Gui m_gui;

    gpu::PushConstants m_pushConstants{};

    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    struct FrameData
    {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        VkFence inFlightFence = VK_NULL_HANDLE;
    };

    std::array<FrameData, kFramesInFlight> m_frames{};
    uint32_t m_frameInFlight = 0;

    struct StorageImage
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    StorageImage m_accumImage{};
    StorageImage m_displayImage{};
    VkExtent2D m_renderExtent{};

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_rendererDescriptorSet = VK_NULL_HANDLE;

    bool createCommandPool();
    bool createCommandBuffers();
    bool createSyncObjects();
    bool createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView& imageView);

    bool createStorageImages();
    bool createStorageImage(VkFormat format, VkImageUsageFlags usage, StorageImage& storageImage);
    void destroyStorageImage(StorageImage& storageImage);

    bool createDescriptorPool();
    bool createRendererDescriptorSet(VkAccelerationStructureKHR TLAS);

    bool recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex);

    static void recordImageBarrier(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldImageLayout,
        VkImageLayout newImageLayout,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess);
};