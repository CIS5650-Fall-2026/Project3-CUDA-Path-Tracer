#pragma once

// Buffer utility functions/structs
// NCHORTEK TODO: StorageImage in Renderer is a good candidate for vkutil as well

#include <volk.h>
#include <vk_mem_alloc.h>

class VulkanContext;

namespace vkutil
{

    struct Buffer
    {
        VkBuffer bufferHandle = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkDeviceSize size = 0;

        // Only populated when VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is set
        VkDeviceAddress deviceAddress = 0;
    };

    // Creates a buffer
    bool createBuffer(
        VulkanContext& context,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkDeviceSize minRequiredAlignment,
        Buffer& buffer);

    // Creates a buffer and fills it from host memory via a staging buffer
    bool createAndUploadBuffer(
        VulkanContext& context,
        const void* data,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        Buffer& buffer);

    // Destroys a buffer. Guaranteed safe to call on a zero-initialized
    // or destroyed Buffer
    void destroyBuffer(VulkanContext& context, Buffer& buffer);

} // namespace vkutil