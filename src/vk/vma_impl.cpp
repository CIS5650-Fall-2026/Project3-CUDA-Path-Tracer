// VMA's implementation must be compiled in exactly one translation unit.
// Keeping it in its own file avoids polluting the codebase with its macros and
// keeps its warnings out of the build output.
//
// VK_NO_PROTOTYPES (from the volk CMake target) removes vulkan.h's function
// declarations, so VMA can't call Vulkan functions by name. Both of its
// function-loading modes are disabled here; volk's pointers are handed to it
// at allocator creation (vmaImportVulkanFunctionsFromVolk)

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION

#include <vk_mem_alloc.h>
