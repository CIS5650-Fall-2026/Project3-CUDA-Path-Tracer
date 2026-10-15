#include "swapchain.h"

#include "context.h"

#include <cstdio>

bool Swapchain::init(VulkanContext& context, uint32_t width, uint32_t height)
{
    m_context = &context;
    return recreate(width, height);
}

void Swapchain::cleanupPerImageResources()
{
    if (m_context == nullptr)
    {
        return;
    }

    // Destroy image views and semaphores
    if (VkDevice device = m_context->getDevice(); device != VK_NULL_HANDLE)
    {
        for (VkImageView imgView : m_imageViews)
        {
            vkDestroyImageView(device, imgView, nullptr);
        }

        for (VkSemaphore semaphore : m_renderFinishedSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    }

    m_renderFinishedSemaphores.clear();
    m_imageViews.clear();
    m_images.clear();
}

bool Swapchain::recreate(uint32_t width, uint32_t height)
{
    if (m_context == nullptr)
    {
        fprintf(stderr, "Attempted to create swapchain without a valid VulkanContext\n");
        return false;
    }

    if (VkDevice device = m_context->getDevice(); device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    vkb::Swapchain oldSwapchain = m_vkbSwapchain;
    cleanupPerImageResources();

    // VK_FORMAT_B8G8R8A8_UNORM is the format every Windows surface supports.
    // UNORM (no hardware sRGB conversion) is what ImGui expects, since it writes
    // sRGB colors as-is.
    // 
    // VK_COLOR_SPACE_SRGB_NONLINEAR_KHR (available on all surfaces) declares that
    // stored values are sRGB-encoded, which we will guarantee by writing gamma-encoded
    // values in our shaders.
    // 
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT is required because we will blit rendered images
    // into the swapchain image prior to presentation.
    auto swapchainResult = vkb::SwapchainBuilder(m_context->getVkbDevice())
        .set_old_swapchain(oldSwapchain)
        .set_desired_format({
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        })
        .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build();

    vkb::destroy_swapchain(oldSwapchain);

    if (!swapchainResult)
    {
        m_vkbSwapchain = {};
        fprintf(stderr, "Swapchain build failed: %s\n", swapchainResult.error().message().c_str());
        return false;
    }

    m_vkbSwapchain = swapchainResult.value();

    auto imagesResult = m_vkbSwapchain.get_images();
    if (!imagesResult)
    {
        fprintf(stderr, "Swapchain get_images failed: %s\n", imagesResult.error().message().c_str());
        destroy();
        return false;
    }

    m_images = std::move(imagesResult.value());

    auto viewsResult = m_vkbSwapchain.get_image_views();
    if (!viewsResult)
    {
        fprintf(stderr, "Swapchain get_image_views failed: %s\n", viewsResult.error().message().c_str());
        destroy();
        return false;
    }

    m_imageViews = std::move(viewsResult.value());

    if (!createRenderFinishedSemaphores())
    {
        fprintf(stderr, "Failed to create render finished semaphores.\n");
        destroy();
        return false;
    }

    return true;
}

void Swapchain::destroy()
{
    // Destroy per-image resources before the swapchain
    cleanupPerImageResources();
    vkb::destroy_swapchain(m_vkbSwapchain);
    m_vkbSwapchain = {};
}

bool Swapchain::createRenderFinishedSemaphores()
{
    m_renderFinishedSemaphores.clear();

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    size_t swapChainSize = m_images.size();
    m_renderFinishedSemaphores.resize(swapChainSize);

    for (size_t i = 0; i < swapChainSize; i++)
    {
        if (vkCreateSemaphore(m_context->getDevice(), &semaphoreInfo, nullptr, &m_renderFinishedSemaphores.at(i))
            != VK_SUCCESS)
        {
            m_renderFinishedSemaphores.at(i) = VK_NULL_HANDLE;
            return false;
        }
    }

    return true;
}
