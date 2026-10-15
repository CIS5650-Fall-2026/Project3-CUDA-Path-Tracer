#pragma once

// Owns the swapchain and its image views. The window is fixed at the scene's
// resolution and non-resizable, so recreate() is only needed for minimize and
// VK_ERROR_OUT_OF_DATE_KHR.

#include <volk.h>
#include <VkBootstrap.h>

#include <cstdint>
#include <vector>

class VulkanContext;

class Swapchain
{
public:
    Swapchain() = default;

    ~Swapchain()
    {
        destroy();
    }

    // Swapchain should be created once and passed by reference as needed,
    // so we should disable copies/moves
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) = delete;
    Swapchain& operator=(Swapchain&&) = delete;

    bool init(VulkanContext& context, uint32_t width, uint32_t height);
    bool recreate(uint32_t width, uint32_t height);
    void destroy();

    VkSwapchainKHR getHandle() const
    {
	    return m_vkbSwapchain.swapchain;
    }

    VkFormat getFormat() const
    {
	    return m_vkbSwapchain.image_format;
    }

    VkExtent2D getExtent() const
    {
	    return m_vkbSwapchain.extent;
    }

    uint32_t getImageCount() const
    {
        return static_cast<uint32_t>(m_images.size());
    }

    VkImage getImage(uint32_t idx) const
    {
        return m_images.at(idx);
    }

    VkImageView getImageView(uint32_t idx) const
    {
        return m_imageViews.at(idx);
    }

    const std::vector<VkImage>& getImages() const
    {
        return m_images;
    }

    const std::vector<VkImageView>& getImageViews() const
    {
        return m_imageViews;
    }

    VkSemaphore getRenderFinishedSemaphore(uint32_t imageIndex) const
    {
        return m_renderFinishedSemaphores.at(imageIndex);
    }

private:
    VulkanContext* m_context = nullptr;
    vkb::Swapchain m_vkbSwapchain{};

    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;

    bool createRenderFinishedSemaphores();
    void cleanupPerImageResources();
};