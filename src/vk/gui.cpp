#include "gui.h"

#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "context.h"
#include "swapchain.h"

#include <cstdio>

bool Gui::init(VulkanContext& context, Swapchain& swapchain, GLFWwindow* window)
{
    // Set up ImGui
    IMGUI_CHECKVERSION();

    ImGui::CreateContext();
    m_imguiContextCreated = true;

    ImGui::StyleColorsLight();
    
    if (!ImGui_ImplGlfw_InitForVulkan(window, true))
    {
        fprintf(stderr, "ImGui_ImplGlfw_InitForVulkan failed.\n");
        destroy();
        return false;
    }

    m_imguiGlfwBackendInitialized = true;

    const VkFormat colorAttachmentFormat = swapchain.getFormat();
    const uint32_t swapchainImageCount = swapchain.getImageCount();

    ImGui_ImplVulkan_InitInfo imguiVkInitInfo{};
    imguiVkInitInfo.ApiVersion = VK_API_VERSION_1_3;
    imguiVkInitInfo.Instance = context.getInstance();
    imguiVkInitInfo.PhysicalDevice = context.getPhysicalDevice();
    imguiVkInitInfo.Device = context.getDevice();
    imguiVkInitInfo.QueueFamily = context.getGraphicsQueueFamily();
    imguiVkInitInfo.Queue = context.getGraphicsQueue();

    // Leaving DescriptorPool null and setting DescriptorPoolSize makes ImGui create
    // and destroy its own pool.
    imguiVkInitInfo.DescriptorPool = VK_NULL_HANDLE;
    imguiVkInitInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;

    // ImGui uses our swapchain, so we match its image count
    imguiVkInitInfo.MinImageCount = swapchainImageCount;
    imguiVkInitInfo.ImageCount = swapchainImageCount;
    imguiVkInitInfo.PipelineCache = VK_NULL_HANDLE;

    // We'll use dynamic rendering, so theres no render pass involved.
    imguiVkInitInfo.UseDynamicRendering = true;
    imguiVkInitInfo.PipelineInfoMain.RenderPass = VK_NULL_HANDLE;
    imguiVkInitInfo.PipelineInfoMain.Subpass = 0;
    imguiVkInitInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    imguiVkInitInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    imguiVkInitInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    imguiVkInitInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorAttachmentFormat;

    imguiVkInitInfo.Allocator = nullptr;
    imguiVkInitInfo.CheckVkResultFn = checkImGuiVkResult;

    if (!ImGui_ImplVulkan_Init(&imguiVkInitInfo))
    {
        fprintf(stderr, "Failed to initialize Vulkan ImGui.\n");
        destroy();
        return false;
    }

    m_imguiVulkanBackendInitialized = true;

    return true;
}

void Gui::checkImGuiVkResult(VkResult result)
{
    if (result != VK_SUCCESS)
    {
        fprintf(stderr, "ImGui Vulkan backend error: VkResult = %d\n", static_cast<int>(result));
    }
}

void Gui::destroy()
{
    if (m_imguiVulkanBackendInitialized)
    {
        ImGui_ImplVulkan_Shutdown();
        m_imguiVulkanBackendInitialized = false;
    }

    if (m_imguiGlfwBackendInitialized)
    {
        ImGui_ImplGlfw_Shutdown();
        m_imguiGlfwBackendInitialized = false;
    }

    if (m_imguiContextCreated)
    {
        ImGui::DestroyContext();
        m_imguiContextCreated = false;
    }
}

void Gui::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Gui::renderFrame()
{
    ImGui::Render();
}

void Gui::recordDrawCommands(VkCommandBuffer commandBuffer, VkImageView swapchainImageView, VkExtent2D swapchainExtent)
{
    // We use a load op rather than clear, because ImGui is rendering on top of
    // an image that HW RT has already filled
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchainImageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = { 0, 0 };
    renderingInfo.renderArea.extent = swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    vkCmdEndRendering(commandBuffer);
}

bool Gui::wantsMouse() const
{
    return ImGui::GetIO().WantCaptureMouse;
}