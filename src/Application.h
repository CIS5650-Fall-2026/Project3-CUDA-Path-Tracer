#pragma once

#include "window.h"
#include "renderer.h"
#include "scene.h"
#include "sceneLoader.h"
#include "camera.h"
#include "inputState.h"
#include "environmentMap.h"

#include <vulkan/vulkan_raii.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

class Application {
public:
	Application();
	~Application();

	Application& operator=(const Application&) = delete;
	Application(const Application&) = delete;

	Application& operator=(Application&&) = delete;
	Application(Application&&) = delete;

	void run();

	Window& getWindow() { return m_window; }

private:
	void InitVulkan();
	void InitCUDAVulkanInterop();
	void InitImGui();

	void recreateSwapchain();

	void renderImGui();

	void pickSceneFile();

	void drawLoadSceneModal();

	void drawFallbackMaterialImGui();

	void saveCurrentRender();

	void pickEnvironmentMap();

	void clearEnvironmentMap();

	vk::raii::Context m_vkContext;
	Window m_window;
	Renderer m_renderer;
	Camera m_camera;
	InputState m_inputState;

	std::unique_ptr<Scene> m_currentScene = nullptr;

	// Load scene modal state
	std::string m_pendingScenePath;
	bool m_bOpenLoadSceneModal = false;
	SceneLoadOptions m_sceneLoadOptions;

	std::unique_ptr<EnvironmentMap> m_currentEnvironmentMap = nullptr;

	vk::raii::Instance m_vkInstance = nullptr;
	vk::raii::SurfaceKHR m_vkSurface = nullptr;
	vk::raii::PhysicalDevice m_vkPhysicalDevice = nullptr;
	vk::raii::Device m_vkDevice = nullptr;
	uint32_t m_graphicsQueueFamilyIndex = 0;
	vk::raii::Queue m_vkGraphicsQueue = nullptr;
	vk::SurfaceFormatKHR m_vkSurfaceFormat;
	vk::Extent2D m_vkSwapchainExtent;
	vk::raii::SwapchainKHR m_vkSwapchain = nullptr;
	std::vector<vk::raii::ImageView> m_vkSwapchainImageViews;

	// CUDA-Vulkan Interop 
	vk::raii::Image m_interopImage = nullptr;
	vk::raii::DeviceMemory m_interopImageMemory = nullptr;
	vk::raii::ImageView m_interopImageView = nullptr;
	HANDLE m_interopImageHandle = nullptr;

	vk::raii::DescriptorPool m_imguiDescriptorPool = nullptr;

	// Command pool + buffer for run loop
	vk::raii::CommandPool m_vkCommandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> m_vkCommandBuffers;

	// Synchronization primitives
	std::vector<vk::raii::Semaphore> m_imageAvailableSemaphores;
	std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
	std::vector<vk::raii::Fence> m_inFlightFences;
	size_t m_currentFrameIndex = 0;

	// Performance tracking
	double m_previousFrameTime = 0.0;
	std::vector<float> m_frameTimeHistory = std::vector<float>(100, 0.0f);
	size_t m_historyOffset = 0;
};