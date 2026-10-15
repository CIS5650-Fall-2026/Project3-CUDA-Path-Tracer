#pragma once

#include <volk.h>

struct GLFWwindow;
class VulkanContext;
class Swapchain;

class Gui
{
public:
    Gui() = default;
    ~Gui()
    {
        destroy();
    }

    // Gui should be created once and passed by reference as needed,
    // so we should disable copies/moves
    Gui(const Gui&) = delete;
    Gui& operator=(const Gui&) = delete;
    Gui(Gui&&) = delete;
    Gui& operator=(Gui&&) = delete;

    bool init(VulkanContext& context, Swapchain& swapchain, GLFWwindow* window);
    void destroy();

    void beginFrame();
    void renderFrame();

    void recordDrawCommands(VkCommandBuffer commandBuffer, VkImageView swapchainImageView, VkExtent2D extent);

    bool wantsMouse() const;

private:
    bool m_imguiContextCreated = false;
    bool m_imguiGlfwBackendInitialized = false;
    bool m_imguiVulkanBackendInitialized = false;

    static void checkImGuiVkResult(VkResult result);
};