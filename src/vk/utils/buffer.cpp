#include "buffer.h"

#include "vk/context.h"

#include <cstdio>
#include <cstring>

namespace vkutil
{
    bool createBufferImpl(
        VulkanContext& context,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VmaAllocationCreateFlags allocFlags,
        VkDeviceSize minRequiredAlignment,
        Buffer& buffer,
        VmaAllocationInfo* allocInfo = nullptr)
    {
        buffer = {};

        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = usage;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocCreateInfo.flags = allocFlags;

        VmaAllocationInfo tmpAllocInfo{};
        const VkResult result = (minRequiredAlignment > 0)
            ? vmaCreateBufferWithAlignment(
                context.getAllocator(), &bufferCreateInfo, &allocCreateInfo, minRequiredAlignment,
                &buffer.bufferHandle, &buffer.allocation, &tmpAllocInfo)
            : vmaCreateBuffer(
                context.getAllocator(), &bufferCreateInfo, &allocCreateInfo,
                &buffer.bufferHandle, &buffer.allocation, &tmpAllocInfo);

        if (result != VK_SUCCESS)
        {
            buffer = {};
            fprintf(stderr, "Failed to create buffer.\n");
            return false;
        }

        buffer.size = size;

        // Set buffer device address only if usage indicates its needed
        if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0)
        {
            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = buffer.bufferHandle;
            buffer.deviceAddress = vkGetBufferDeviceAddress(context.getDevice(), &addressInfo);
        }

        if (allocInfo != nullptr)
        {
            *allocInfo = tmpAllocInfo;
        }

        return true;
    }

    bool createBuffer(
        VulkanContext& context,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkDeviceSize minRequiredAlignment,
        Buffer& buffer)
    {
        return createBufferImpl(context, size, usage, 0, minRequiredAlignment, buffer);
    }

    bool createAndUploadBuffer(
        VulkanContext& context,
        const void* data,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        Buffer& buffer)
    {
        if (data == nullptr || size == 0)
        {
            fprintf(stderr, "createAndUploadBuffer called with null or empty data.\n");
            return false;
        }

        buffer = {};

        // Use a host-visible, mapped staging buffer so the final buffer
        // can be optimized for GPU reads
        Buffer stagingBuffer{};
        VmaAllocationInfo stagingAllocInfo{};
        if (!createBufferImpl(
            context,
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            0,
            stagingBuffer,
            &stagingAllocInfo))
        {
            return false;
        }

        // Copy the CPU data into the staging buffer
        memcpy(stagingAllocInfo.pMappedData, data, static_cast<size_t>(size));

        // Make the writes visible to the device in case the memory isn't host-coherent
        // (host-coherent memory is already GPU-visible, so it'd be a no-op in that case)
        if (vmaFlushAllocation(context.getAllocator(), stagingBuffer.allocation, 0, VK_WHOLE_SIZE)
            != VK_SUCCESS)
        {
            fprintf(stderr, "Failed to flush staging buffer memory.\n");
            destroyBuffer(context, stagingBuffer);
            return false;
        }

        if (!createBufferImpl(context, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, 0, buffer))
        {
            destroyBuffer(context, stagingBuffer);
            return false;
        }

        VkCommandBuffer commandBuffer = context.beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, stagingBuffer.bufferHandle, buffer.bufferHandle, 1, &copyRegion);

        context.endSingleTimeCommands(commandBuffer);

        destroyBuffer(context, stagingBuffer);
        return true;
    }

    void destroyBuffer(VulkanContext& context, Buffer& buffer)
    {
        // vmaDestroyBuffer ignores VK_NULL_HANDLE, so this is safe on an unused Buffer.
        vmaDestroyBuffer(context.getAllocator(), buffer.bufferHandle, buffer.allocation);
        buffer = {};
    }

} // namespace vkutil