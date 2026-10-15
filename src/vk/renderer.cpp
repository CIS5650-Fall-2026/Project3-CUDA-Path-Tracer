#include "renderer.h"

#include "context.h"
#include "swapchain.h"
#include "gpuscene.h"
#include "gpu/shared.h"

#include <array>
#include <cstdio>

#include <VkBootstrap.h>

bool Renderer::init(VulkanContext& context, Swapchain& swapchain, GLFWwindow* window, uint32_t width, uint32_t height, GpuScene& gpuScene)
{
    m_context = &context;
    m_swapchain = &swapchain;
    m_renderExtent.width = width;
    m_renderExtent.height = height;

    // These device address pointers never change so its safe to just set them
    // once on init rather than per-frame
    m_pushConstants.sceneAddr = gpuScene.getSceneAddresses();
    m_pushConstants.renderedFrameCount = 0;

    if (!m_gui.init(context, swapchain, window))
    {
        destroy();
        return false;
    }

    if (!m_pipeline.init(context))
    {
        destroy();
        return false;
    }

    if (!createCommandPool())
    {
        destroy();
        return false;
    }

    if (!createCommandBuffers())
    {
        destroy();
        return false;
    }

    if (!createSyncObjects())
    {
        destroy();
        return false;
    }

    if (!createStorageImages())
    {
        destroy();
        return false;
    }

    if (!createDescriptorPool())
    {
        destroy();
        return false;
    }

    if (!createRendererDescriptorSet(gpuScene.getTLAS()))
    {
        destroy();
        return false;
    }

    return true;
}

void Renderer::destroy()
{
    if (m_context == nullptr)
    {
        return;
    }

    VkDevice device = m_context->getDevice();
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    // Destroy in reverse order of creation
    // storage images --> sync objects --> command pool (automatically destroys command buffers)
    destroyStorageImage(m_displayImage);
    destroyStorageImage(m_accumImage);

    for (FrameData& frameData : m_frames)
    {
        frameData.commandBuffer = VK_NULL_HANDLE;

        vkDestroySemaphore(device, frameData.imageAvailableSemaphore, nullptr);
        frameData.imageAvailableSemaphore = VK_NULL_HANDLE;

        vkDestroyFence(device, frameData.inFlightFence, nullptr);
        frameData.inFlightFence = VK_NULL_HANDLE;
    }

    vkDestroyCommandPool(device, m_commandPool, nullptr);
    m_commandPool = VK_NULL_HANDLE;

    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    m_descriptorPool = VK_NULL_HANDLE;
    m_rendererDescriptorSet = VK_NULL_HANDLE;

    m_frames = {};
    m_frameInFlight = 0;
    m_pipeline.destroy();
    m_gui.destroy();
    m_pushConstants = {};
    m_swapchain = nullptr;
    m_context = nullptr;
}

bool Renderer::drawFrame()
{
    VkDevice device = m_context->getDevice();
    VkSwapchainKHR swapchain = m_swapchain->getHandle();
    FrameData& frameData = m_frames.at(m_frameInFlight);

    // Wait for an earlier frame to finish rendering before kicking off a new one,
    // by specifying the inFlightFence
    int fenceCount = 1;
    vkWaitForFences(device, fenceCount, &frameData.inFlightFence, VK_TRUE, kMaxTimeout);

    // Acquire an image from the swapchain, and signal imageAvailableSemaphore when this
    // command completes
    uint32_t availableImageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device,
        swapchain,
        kMaxTimeout,
        frameData.imageAvailableSemaphore,
        VK_NULL_HANDLE,
        &availableImageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        // The swap chain has become incompatible with the surface and can no longer
        // be used for rendering (it becomes impossible to present).
        // This usually happens after a window minimzation/resize--recreate the swap chain
        // and then try rendering again.
        return m_swapchain->recreate(m_renderExtent.width, m_renderExtent.height);
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        // We exclude VK_SUBOPTIMAL_KHR here because that means the swap chain can still be
        // used to successfully present to the surface, even though the surface properties are
        // no longer matched exactly. All other result values indicate a critical image-acquisition
        // failure.
        fprintf(stderr, "Failed to acquire swap chain image.\n");
        return false;
    }

    // After waiting, we need to manually reset the fence to the unsignaled state.
    // Crucially, we only reset the fence once we know we will be submitting new work
    // (after we have verified that the swapchain is valid and we have successfully
    // acquired an available image to render to)
    vkResetFences(device, fenceCount, &frameData.inFlightFence);

    // Record the command buffer, setting it up with all the required rendering commands
    if (!this->recordCommandBuffer(frameData.commandBuffer, availableImageIndex))
    {
        return false;
    }

    // Prepare to submit our command buffer!
    // Instruct the GPU to wait for an available image before executing the
    // blit-stage commands in the command buffer. This stage must match the
    // swapchain barrier's srcStage (blit), so the layout transition happens
    // after the acquire.
    VkSemaphoreSubmitInfo waitSemaphoreInfo{};
    waitSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemaphoreInfo.semaphore = frameData.imageAvailableSemaphore;
    waitSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;

    // Specify which semaphore to signal once the command buffer has finished execution.
    // VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT ensures that the semaphore is signaled only
    // after every command in the buffer has completed.
    VkSemaphoreSubmitInfo signalSemaphoreInfo{};
    signalSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphoreInfo.semaphore = m_swapchain->getRenderFinishedSemaphore(availableImageIndex);
    signalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = frameData.commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemaphoreInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemaphoreInfo;

    // Submit this frame's command buffer to the graphics queue, and tell the GPU
    // to signal inFlightFence when the command buffer finishes execution
    if (vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &submitInfo, frameData.inFlightFence)
        != VK_SUCCESS)
    {
        fprintf(stderr, "Failed to submit the frame command buffer.\n");
        return false;
    }

    // Only increment this after we know that we successfully submitted this frame's
    // command buffer
    m_pushConstants.renderedFrameCount++;

    // Now that we have told the GPU to render a frame, we need to actually
    // handle presentation of that frame to the screen. Presentation can't
    // occur until rendering finishes, so wait on the appropriate semaphore
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &signalSemaphoreInfo.semaphore;

    // Specify the swap chains to present images to and the index of the
    // image for each swap chain. There is typically just one swapchain
    VkSwapchainKHR swapChains[] = { swapchain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &availableImageIndex;
    presentInfo.pResults = nullptr;

    // Finally, present our rendered frame to the screen!
    result = vkQueuePresentKHR(m_context->getGraphicsQueue(), &presentInfo);

    // Recreate the swap chain if it has become invalid/suboptimal
    if (result == VK_ERROR_OUT_OF_DATE_KHR
        || result == VK_SUBOPTIMAL_KHR)
    {
        // VK_SUBOPTIMAL_KHR is considered a successful result (i.e. the image was
        // successfully presented), but we still should try to fix the swapchain.
        if (!m_swapchain->recreate(m_renderExtent.width, m_renderExtent.height))
        {
            return false;
        }
    }
    else if (result != VK_SUCCESS)
    {
        fprintf(stderr, "Failed to present the swap chain image.\n");
        return false;
    }

    // Increment the current frame index and ensure it loops around after
    // every kFramesInFlight enqueued frames.
    m_frameInFlight = (m_frameInFlight + 1) % kFramesInFlight;
    return true;
}

bool Renderer::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

    // We will be recording a command buffer every frame, so we want to be
    // able to reset and rerecord over it. This flag allows command buffers
    // to be rerecorded individually, without this flag they all have to be
    // reset together.
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    // Each command pool can only allocate command buffers that are submitted
    // on a single type of queue. We will use command buffers for rendering.
    poolInfo.queueFamilyIndex = m_context->getGraphicsQueueFamily();

    if (vkCreateCommandPool(m_context->getDevice(), &poolInfo, nullptr, &m_commandPool)
        != VK_SUCCESS)
    {
        m_commandPool = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create command pool.\n");
        return false;
    }

    return true;
}

bool Renderer::createCommandBuffers()
{
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = m_commandPool;

    // Primary command buffers can be submitted to a queue for execution,
    // but cannot be called from other command buffers.
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkDevice device = m_context->getDevice();
    for (FrameData& frameData : m_frames)
    {
        // Note that command buffers are automatically deallocated when the command pool that
        // created them is destroyed, so we don't need do any manual cleanup
        if (vkAllocateCommandBuffers(device, &allocateInfo, &frameData.commandBuffer)
            != VK_SUCCESS)
        {
            fprintf(stderr, "Failed to allocate command buffers.\n");
            return false;
        }
    }

    return true;
}

bool Renderer::createSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    // Create the fence in the signaled state, so that the first call to vkWaitForFences()
    // returns immediately since the fence is already signaled.
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkDevice device = m_context->getDevice();
    for (FrameData& frameData : m_frames)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frameData.imageAvailableSemaphore)
            != VK_SUCCESS)
        {
            frameData.imageAvailableSemaphore = VK_NULL_HANDLE;
            fprintf(stderr, "Failed to create image-available semaphore.\n");
            return false;
        }

        if (vkCreateFence(device, &fenceInfo, nullptr, &frameData.inFlightFence) != VK_SUCCESS)
        {
            frameData.inFlightFence = VK_NULL_HANDLE;
            fprintf(stderr, "Failed to create in-flight fence.\n");
            return false;
        }
    }

    return true;
}

bool Renderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView& imageView)
{
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image;

    // Specify how the image data should be interpreted
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format;

    // // Use default color channel mapping
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    // Describe the image purpose and specify which part of the image
    // should be accessed
    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;

    // Multiple array layers are only needed for stereographic 3D applications,
    // allowing different views to be displayed for each eye. A traditional
    // renderer only needs 1 layer.
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_context->getDevice(), &createInfo, nullptr, &imageView)
        != VK_SUCCESS)
    {
        imageView = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create image view.\n");
        return false;
    }

    return true;
}

bool Renderer::createStorageImage(VkFormat format, VkImageUsageFlags usage, StorageImage& storageImage)
{
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = {
        m_renderExtent.width,
        m_renderExtent.height,
        1
    };
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    if (vmaCreateImage(m_context->getAllocator(), &imageCreateInfo, &allocCreateInfo, &storageImage.image, &storageImage.allocation, nullptr)
        != VK_SUCCESS)
    {
        storageImage = {};
        fprintf(stderr, "Failed to create VkImage for storage image.\n");
        return false;
    }

    storageImage.format = format;

    if (!createImageView(storageImage.image, format, VK_IMAGE_ASPECT_COLOR_BIT, storageImage.imageView))
    {
        return false;
    }

    return true;
}

void Renderer::destroyStorageImage(StorageImage& storageImage)
{
    // Destroy in reverse order of creation
    vkDestroyImageView(m_context->getDevice(), storageImage.imageView, nullptr);
    vmaDestroyImage(m_context->getAllocator(), storageImage.image, storageImage.allocation);
    storageImage = {};
}

bool Renderer::createStorageImages()
{
    if (!createStorageImage(
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        m_accumImage))
    {
        return false;
    }

    if (!createStorageImage(
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        m_displayImage))
    {
        return false;
    }

    VkCommandBuffer commandBuffer = m_context->beginSingleTimeCommands();

    std::array<VkImage, 2> storageImages = {
        m_accumImage.image,
        m_displayImage.image
    };

    for (VkImage image : storageImages)
    {
        // Transition each storage image to VK_IMAGE_LAYOUT_GENERAL so shaders have read/write access.
        recordImageBarrier(
            commandBuffer,
            image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    }

    m_context->endSingleTimeCommands(commandBuffer);

    return true;
}

bool Renderer::createDescriptorPool()
{
    // One pool size per descriptor type, counting descriptors across all sets
    std::array<VkDescriptorPoolSize, 2> poolSizes{};

    // One for the accumulation image and the display image
    poolSizes.at(0).type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes.at(0).descriptorCount = 2;

    // One for our TLAS
    poolSizes.at(1).type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes.at(1).descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(m_context->getDevice(), &poolInfo, nullptr, &m_descriptorPool)
        != VK_SUCCESS)
    {
        m_descriptorPool = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create descriptor pool.\n");
        return false;
    }

    return true;
}

bool Renderer::createRendererDescriptorSet(VkAccelerationStructureKHR TLAS)
{
    VkDevice device = m_context->getDevice();
    VkDescriptorSetLayout layout = m_pipeline.getRendererDescriptorSetLayout();

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &m_rendererDescriptorSet)
        != VK_SUCCESS)
    {
        m_rendererDescriptorSet = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to allocate descriptor sets.\n");
        return false;
    }

    // Unlike other descriptor writes, writing a TLAS must be done via
    // a VkWriteDescriptorSet pointing to a VkWriteDescriptorSetAccelerationStructureKHR
    // struct.
    VkWriteDescriptorSetAccelerationStructureKHR descWriteTLAS{};
    descWriteTLAS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    descWriteTLAS.accelerationStructureCount = 1;
    descWriteTLAS.pAccelerationStructures = &TLAS;

    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    accumImageInfo.imageView = m_accumImage.imageView;

    VkDescriptorImageInfo displayImageInfo{};
    displayImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    displayImageInfo.imageView = m_displayImage.imageView;

    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

    descriptorWrites.at(0).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites.at(0).dstSet = m_rendererDescriptorSet;
    descriptorWrites.at(0).dstBinding = gpu::kTLASBinding;
    descriptorWrites.at(0).descriptorCount = 1;
    descriptorWrites.at(0).descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    descriptorWrites.at(0).pNext = &descWriteTLAS;

    descriptorWrites.at(1).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites.at(1).dstSet = m_rendererDescriptorSet;
    descriptorWrites.at(1).dstBinding = gpu::kAccumImageBinding;
    descriptorWrites.at(1).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites.at(1).descriptorCount = 1;
    descriptorWrites.at(1).pImageInfo = &accumImageInfo;

    descriptorWrites.at(2).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites.at(2).dstSet = m_rendererDescriptorSet;
    descriptorWrites.at(2).dstBinding = gpu::kDisplayImageBinding;
    descriptorWrites.at(2).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites.at(2).descriptorCount = 1;
    descriptorWrites.at(2).pImageInfo = &displayImageInfo;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    return true;
}

bool Renderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    // Only relevant for secondary command buffers, leave as null
    beginInfo.pInheritanceInfo = nullptr;

    // If the command buffer was already recorded once, then a call to
    // vkBeginCommandBuffer will implicitly reset it. It's not possible
    // to append commands to a buffer at a later time. This makes
    // manual resetting unnecessary.
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        fprintf(stderr, "Failed to begin recording command buffer.\n");
        return false;
    }

    // Ensure that raygen does not read or write to the accumulation image
    // until the previous frame has finished writing to it.
    recordImageBarrier(
        commandBuffer,
        m_accumImage.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Ensure that raygen does not write to the display image
    // until the previous frame's blit has finished reading it
    recordImageBarrier(
        commandBuffer,
        m_displayImage.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_BLIT_BIT,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Specify which pipeline to use
    vkCmdBindPipeline(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        m_pipeline.getHandle());

    // Specify which descriptor sets to use
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        m_pipeline.getPipelineLayout(),
        gpu::kRendererDescriptorSet,
        1,
        &m_rendererDescriptorSet,
        0,
        nullptr);

    // Update our push constants
    vkCmdPushConstants(
        commandBuffer,
        m_pipeline.getPipelineLayout(),
        Pipeline::kPushConstantStages,
        0,
        sizeof(m_pushConstants),
        &m_pushConstants);

    // Trace those rays!
    vkCmdTraceRaysKHR(
        commandBuffer,
        &m_pipeline.getRaygenDeviceAddrRegion(),
        &m_pipeline.getMissDeviceAddrRegion(),
        &m_pipeline.getHitDeviceAddrRegion(),
        &m_pipeline.getCallableDeviceAddrRegion(),
        m_renderExtent.width,
        m_renderExtent.height,
        1);

    // Ensure that raygen's display image writes finish before the blit reads them
    recordImageBarrier(
        commandBuffer,
        m_displayImage.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BLIT_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT);

    const VkImage swapchainImage = m_swapchain->getImage(swapchainImageIndex);
    const VkExtent2D swapchainExtent = m_swapchain->getExtent();

    // Ensure that the swapchain image transitions to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    // and that the transition occurs after the submit's blit-stage wait on the imageAvailable 
    // semaphore completes. This guarantees that the swapchain image is done being used by the
    // swapchain for presentation. Additionally, ensure that this frame's blit does not begin
    // until after the layout transition occurs.
    recordImageBarrier(
        commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_BLIT_BIT,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_BLIT_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Blit the display image to our swapchain image. This automatically handles any relevant
    // image format conversions
    VkImageBlit imageBlit{};
    imageBlit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    imageBlit.srcOffsets[1] = {
        static_cast<int32_t>(m_renderExtent.width),
        static_cast<int32_t>(m_renderExtent.height),
        1
    };
    imageBlit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    imageBlit.dstOffsets[1] = {
        static_cast<int32_t>(swapchainExtent.width),
        static_cast<int32_t>(swapchainExtent.height),
        1
    };

    vkCmdBlitImage(
        commandBuffer,
        m_displayImage.image,
        VK_IMAGE_LAYOUT_GENERAL,
        swapchainImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &imageBlit,
        VK_FILTER_NEAREST);

    // Transition the swapchain image into color attachment layout after the blit
    // writes are visible so ImGui can draw on top of it
    recordImageBarrier(
        commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_BLIT_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    m_gui.recordDrawCommands(
        commandBuffer,
        m_swapchain->getImageView(swapchainImageIndex),
        swapchainExtent);

    // Transition into presentation layout once ImGui's writes are visible
    recordImageBarrier(
        commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        fprintf(stderr, "Failed to end recording to command buffer.\n");
        return false;
    }

    return true;
}

void Renderer::recordImageBarrier(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldImageLayout,
    VkImageLayout newImageLayout,
    VkPipelineStageFlags2 srcStage,
    VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,
    VkAccessFlags2 dstAccess)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldImageLayout;
    barrier.newLayout = newImageLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}