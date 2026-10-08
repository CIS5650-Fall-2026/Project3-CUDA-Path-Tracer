#include "application.h"

#include "glfw/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "stb_image.h"

#include "nfd.hpp"

#include <cuda_runtime.h>

#include <iostream>

#include "kernel.h"

Application::Application() {
	if (!glfwInit()) {
		throw std::runtime_error("Failed to initialize GLFW");
	}

	m_window = Window(800, 600, "CUDA-Vulkan Path Tracer");

	glfwSetWindowUserPointer(m_window.GetGLFWwindow(), this);

	glfwSetKeyCallback(m_window.GetGLFWwindow(), [](GLFWwindow* window, int key, int scancode, int action, int mods) {
		auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
		if (!app) {
			return;
		}

		if (action == GLFW_PRESS) {
			app->m_inputState.setKeyPressed(key, true);
		}
		if (action == GLFW_RELEASE) {
			app->m_inputState.setKeyPressed(key, false);
		}
		});

	glfwSetMouseButtonCallback(m_window.GetGLFWwindow(), [](GLFWwindow* window, int button, int action, int mods) {
		auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
		if (!app) {
			return;
		}

		if (action == GLFW_PRESS) {
			app->m_inputState.setMouseButtonPressed(button, true);
		}
		if (action == GLFW_RELEASE) {
			app->m_inputState.setMouseButtonPressed(button, false);
		}
		});

	glfwSetCursorPosCallback(m_window.GetGLFWwindow(), [](GLFWwindow* window, double xpos, double ypos) {
		auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
		if (!app) {
			return;
		}

		if (app->m_inputState.isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT)) {
			glm::vec2 deltaMousePosition = glm::vec2(xpos, ypos) - app->m_inputState.m_mousePosition;
			app->m_camera.rotate(deltaMousePosition);
		}

		app->m_inputState.m_mousePosition = glm::vec2(xpos, ypos);


		});

	glfwSetScrollCallback(m_window.GetGLFWwindow(), [](GLFWwindow* window, double xoffset, double yoffset) {
		auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
		if (!app) {
			return;
		}
		app->m_camera.setScrollOffset(static_cast<float>(yoffset));
		});


	InitVulkan();

	InitCUDAVulkanInterop();
		
	InitImGui();

	m_previousFrameTime = glfwGetTime();

	m_currentScene = SceneLoader::loadFromFile("scenes/cornell.json");
}

Application::~Application() {
	ImGui_ImplVulkan_Shutdown();

	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();
}

void Application::run() {
	size_t maxFramesInFlight = m_vkSwapchainImageViews.size();

	while (!glfwWindowShouldClose(m_window.GetGLFWwindow())) {
		glfwPollEvents();

		if (m_window.m_hasBeenResized) {
			recreateSwapchain();
			m_window.m_hasBeenResized = false;
			continue;
		}

		// Update performance metrics
		double currentTime = glfwGetTime();
		float deltaTime = static_cast<float>(currentTime - m_previousFrameTime);
		m_previousFrameTime = currentTime;

		float currentFps = 0.0f;
		float currentFrameTimeMs = 0.0f;
		if (deltaTime > 0.0f) {
			currentFps = 1.0f / deltaTime;
			currentFrameTimeMs = deltaTime * 1000.f;
		}

		// Add to circular buffer
		m_frameTimeHistory[m_historyOffset] = currentFrameTimeMs;
		m_historyOffset = (m_historyOffset + 1) % m_frameTimeHistory.size();

		// Get current objects in flight
		auto& currentInFlightFence = m_inFlightFences[m_currentFrameIndex];
		auto& currentImageAvailableSemaphore = m_imageAvailableSemaphores[m_currentFrameIndex];

		auto& currentCommandBuffer = m_vkCommandBuffers[m_currentFrameIndex];

		// Wait for GPU to finish
		while (m_vkDevice.waitForFences(
			{ *currentInFlightFence },
			VK_TRUE,
			UINT64_MAX
		) == vk::Result::eTimeout);
		m_vkDevice.resetFences({ *currentInFlightFence });

		// Grab next swapchain image
		vk::Result acquireResult = vk::Result::eSuccess;
		uint32_t imageIndex = 0;
		try {
			std::tie(acquireResult, imageIndex) = m_vkSwapchain.acquireNextImage(UINT64_MAX, *currentImageAvailableSemaphore);
		}
		catch (const vk::OutOfDateKHRError&) {
			recreateSwapchain();
			continue;
		}

		if (acquireResult == vk::Result::eSuboptimalKHR) {
			recreateSwapchain();
			continue;
		}
		else if (acquireResult != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to acquire swapchain iamge");
		}

		// Use render finished semaphore from acquired image
		auto& currentRenderFinishedSemaphore = m_renderFinishedSemaphores[imageIndex];

		// Do ImGui Rendering
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		renderImGui();

		ImGui::Render();

		// Update camera
		m_camera.tick(deltaTime, m_inputState);

		// Render scene
		m_renderer.render(m_currentScene, m_currentEnvironmentMap, m_camera, m_camera.m_hasChanged);

		// Clear camera has moved
		m_camera.m_hasChanged = false;

		currentCommandBuffer.reset();
		currentCommandBuffer.begin(vk::CommandBufferBeginInfo{});

		// Get raw swapchain image
		std::vector<vk::Image> rawSwapchainImages = m_vkSwapchain.getImages();
		vk::Image activeImage = rawSwapchainImages[imageIndex];

		// Set up barrier for copying interop image
		vk::ImageMemoryBarrier interopBarrier{};
		interopBarrier.oldLayout = vk::ImageLayout::eUndefined;
		interopBarrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
		interopBarrier.image = *m_interopImage;
		interopBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		interopBarrier.subresourceRange.levelCount = 1;
		interopBarrier.subresourceRange.layerCount = 1;
		interopBarrier.srcAccessMask = vk::AccessFlags(0);
		interopBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

		// Set up barrier for writing to swapchain image
		vk::ImageMemoryBarrier swapchainBarrier{};
		swapchainBarrier.oldLayout = vk::ImageLayout::eUndefined;
		swapchainBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
		swapchainBarrier.image = activeImage;
		swapchainBarrier.subresourceRange = interopBarrier.subresourceRange;
		swapchainBarrier.srcAccessMask = vk::AccessFlags(0);
		swapchainBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

		currentCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			vk::DependencyFlags(0),
			nullptr,
			nullptr,
			{ interopBarrier, swapchainBarrier }
		);

		// Copy interop texture to swapchain
		vk::ImageCopy copyRegion{};
		copyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		copyRegion.srcSubresource.layerCount = 1;
		copyRegion.dstSubresource = copyRegion.srcSubresource;
		copyRegion.extent = vk::Extent3D(m_vkSwapchainExtent.width, m_vkSwapchainExtent.height, 1);

		currentCommandBuffer.copyImage(
			*m_interopImage, 
			vk::ImageLayout::eTransferSrcOptimal,
			activeImage, 
			vk::ImageLayout::eTransferDstOptimal,
			{ copyRegion }
		);

		// Set up barrier to write to image
		vk::ImageMemoryBarrier barrierToRender{};
		barrierToRender.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		barrierToRender.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
		barrierToRender.image = activeImage;
		barrierToRender.subresourceRange = interopBarrier.subresourceRange;
		barrierToRender.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrierToRender.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

		currentCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::DependencyFlags(0), 
			nullptr, 
			nullptr, 
			{ barrierToRender }
		);
		
		// Begin draw
		vk::RenderingAttachmentInfo colorAttachment{};
		colorAttachment.imageView = *m_vkSwapchainImageViews[imageIndex];
		colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
		colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;

		vk::RenderingInfo renderingInfo{};
		renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
		renderingInfo.renderArea.extent = m_vkSwapchainExtent;
		renderingInfo.layerCount = 1;
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &colorAttachment;
	
		currentCommandBuffer.beginRendering(renderingInfo);

		// Add imgui draw data
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *currentCommandBuffer);

		currentCommandBuffer.endRendering();

		// Barrier for image to go back to present mode
		vk::ImageMemoryBarrier barrierToPresent = barrierToRender;
		barrierToPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
		barrierToPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
		barrierToPresent.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		barrierToPresent.dstAccessMask = vk::AccessFlags(0);

		currentCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::PipelineStageFlagBits::eBottomOfPipe,
			vk::DependencyFlags(0), nullptr, nullptr, { barrierToPresent }
		);

		currentCommandBuffer.end();

		// Submit to queue
		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

		vk::SubmitInfo submitInfo{};
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &(*currentImageAvailableSemaphore);
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &(*currentCommandBuffer);
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &(*currentRenderFinishedSemaphore);

		m_vkGraphicsQueue.submit({ submitInfo }, *currentInFlightFence);

		// Present the image
		vk::PresentInfoKHR presentInfo{};
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &(*currentRenderFinishedSemaphore);
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &(*m_vkSwapchain);
		presentInfo.pImageIndices = &imageIndex;

		try {
			vk::Result presentResult = m_vkGraphicsQueue.presentKHR(presentInfo);
		
			if (presentResult == vk::Result::eSuboptimalKHR) {
				recreateSwapchain();
			}
		}
		catch (const vk::OutOfDateKHRError&) {
			recreateSwapchain();
		} 

		m_currentFrameIndex = (m_currentFrameIndex + 1) % maxFramesInFlight;
	}

	// Wait for device completion before exiting
	m_vkDevice.waitIdle();
}

void Application::InitVulkan() {
	vk::ApplicationInfo appInfo{};
	appInfo.pApplicationName = "CUDA-Vulkan Path Tracer";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = vk::ApiVersion13;

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	vk::InstanceCreateInfo createInfo{};
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = glfwExtensionCount;
	createInfo.ppEnabledExtensionNames = glfwExtensions;

	vk::raii::Instance tempVkInstance(m_vkContext, createInfo);
	m_vkInstance = std::move(tempVkInstance);

	// Pick physical device
	vk::raii::PhysicalDevices devices(m_vkInstance);
	if (devices.empty()) {
		throw std::runtime_error("Failed to find GPUs with Vulkan support!");
	}

	for (const auto& device : devices) {
		vk::PhysicalDeviceProperties properties = device.getProperties();
		if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
			m_vkPhysicalDevice = device;
		}
	}

	if (m_vkPhysicalDevice == nullptr) {
		std::cout << "NVIDIA Discrete GPU not found. Falling back to first available device.\n";
		m_vkPhysicalDevice = devices[0];
	}

	// Set up vulkan surface
	VkSurfaceKHR rawSurface;
	if (glfwCreateWindowSurface(*m_vkInstance, m_window.GetGLFWwindow(), nullptr, &rawSurface) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create window surface!");
	}
	m_vkSurface = vk::raii::SurfaceKHR(m_vkInstance, rawSurface);

	// Select a queue family
	auto queueFamilies = m_vkPhysicalDevice.getQueueFamilyProperties();
	bool foundSuitableQueue = false;

	for(uint32_t i = 0; i < queueFamilies.size(); ++i) {
		bool supportsGraphics = (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags(0);
		bool supportsPresent = m_vkPhysicalDevice.getSurfaceSupportKHR(i, *m_vkSurface);
		if(supportsGraphics && supportsPresent) {
			m_graphicsQueueFamilyIndex = i;
			foundSuitableQueue = true;
			break;
		}
	}
	if(!foundSuitableQueue) {
		throw std::runtime_error("Failed to find a suitable queue family!");
	}

	// Create physical device
	float queuePriority = 1.0f;
	vk::DeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
	};

	vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
	dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

	vk::DeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.pNext = &dynamicRenderingFeatures;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

	vk::raii::Device tempVkDevice(m_vkPhysicalDevice, deviceCreateInfo);
	m_vkDevice = std::move(tempVkDevice);

	// Retrieve queue from device
	m_vkGraphicsQueue = vk::raii::Queue(m_vkDevice, m_graphicsQueueFamilyIndex, 0);

	// Create swapchain
	vk::SurfaceCapabilitiesKHR capabilities = m_vkPhysicalDevice.getSurfaceCapabilitiesKHR(*m_vkSurface);
	auto formats = m_vkPhysicalDevice.getSurfaceFormatsKHR(*m_vkSurface);
	auto presentModes = m_vkPhysicalDevice.getSurfacePresentModesKHR(*m_vkSurface);

	m_vkSurfaceFormat = formats[0];
	for(const auto& format : formats) {
		if(format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			m_vkSurfaceFormat = format;
			break;
		}
	}

	vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
	for(const auto& mode : presentModes) {
		if(mode == vk::PresentModeKHR::eMailbox) {
			presentMode = mode;
			break;
		}
	}

	m_vkSwapchainExtent = capabilities.currentExtent;
	if (m_vkSwapchainExtent.width == UINT32_MAX) {
		// Fallback to window size
		m_vkSwapchainExtent.width = 800;
		m_vkSwapchainExtent.height = 600;
	}

	uint32_t imageCount = capabilities.minImageCount + 1;
	if(capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
		imageCount = capabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR swapchainCreateInfo{};
	swapchainCreateInfo.surface = *m_vkSurface;
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = m_vkSurfaceFormat.format;
	swapchainCreateInfo.imageColorSpace = m_vkSurfaceFormat.colorSpace;
	swapchainCreateInfo.imageExtent = m_vkSwapchainExtent;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
	swapchainCreateInfo.preTransform = capabilities.currentTransform;
	swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainCreateInfo.presentMode = presentMode;
	swapchainCreateInfo.clipped = VK_TRUE;

	vk::raii::SwapchainKHR tempVkSwapchain(m_vkDevice, swapchainCreateInfo);
	m_vkSwapchain = std::move(tempVkSwapchain);

	// Get swapchain image views
	std::vector<vk::Image> swapchainImages = m_vkSwapchain.getImages();

	m_vkSwapchainImageViews.clear();
	m_vkSwapchainImageViews.reserve(swapchainImages.size());

	for (const auto& image : swapchainImages) {
		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.image = image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = m_vkSurfaceFormat.format;
		viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
		viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
		viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
		viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_vkSwapchainImageViews.emplace_back(m_vkDevice, viewInfo);
	}

	// Create command pool
	vk::CommandPoolCreateInfo poolInfo{};
	poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	poolInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
	m_vkCommandPool = vk::raii::CommandPool(m_vkDevice, poolInfo);

	size_t maxFramesInFlight = m_vkSwapchainImageViews.size();

	// Allocate command buffer
	vk::CommandBufferAllocateInfo allocInfo{};
	allocInfo.commandPool = *m_vkCommandPool;
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = static_cast<uint32_t>(maxFramesInFlight);

	vk::raii::CommandBuffers buffers(m_vkDevice, allocInfo);
	m_vkCommandBuffers = std::move(buffers);

	// Create synchronization primitives
	m_imageAvailableSemaphores.clear();
	m_renderFinishedSemaphores.clear();
	m_inFlightFences.clear();

	vk::SemaphoreCreateInfo semaphoreInfo{};
	vk::FenceCreateInfo fenceInfo{};
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	for (size_t i = 0; i < maxFramesInFlight; ++i) {
		m_imageAvailableSemaphores.emplace_back(m_vkDevice, semaphoreInfo);
		m_renderFinishedSemaphores.emplace_back(m_vkDevice, semaphoreInfo);
		m_inFlightFences.emplace_back(m_vkDevice, fenceInfo);
	}
}

void Application::InitCUDAVulkanInterop()
{
	vk::ExternalMemoryImageCreateInfo externalMemoryImageInfo{};
	externalMemoryImageInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

	vk::ImageCreateInfo imageInfo{};
	imageInfo.pNext = &externalMemoryImageInfo;
	imageInfo.imageType = vk::ImageType::e2D;
	imageInfo.format = m_vkSurfaceFormat.format;
	imageInfo.extent = vk::Extent3D(m_vkSwapchainExtent.width, m_vkSwapchainExtent.height, 1);
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = vk::SampleCountFlagBits::e1;
	imageInfo.tiling = vk::ImageTiling::eOptimal;
	imageInfo.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	imageInfo.sharingMode = vk::SharingMode::eExclusive;
	imageInfo.initialLayout = vk::ImageLayout::eUndefined;

	m_interopImage = vk::raii::Image(m_vkDevice, imageInfo);

	// Get memory requirements for the interop image
	vk::MemoryRequirements memoryRequirements = m_interopImage.getMemoryRequirements();

	// Configure export handle for win32
	vk::ExportMemoryAllocateInfo exportAllocateInfo{};
	exportAllocateInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

	vk::MemoryAllocateInfo allocateInfo{};
	allocateInfo.allocationSize = memoryRequirements.size;
	allocateInfo.pNext = &exportAllocateInfo;

	// Find a memory type
	auto memProperties = m_vkPhysicalDevice.getMemoryProperties();
	uint32_t memTypeIndex = UINT32_MAX;
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
		if ((memoryRequirements.memoryTypeBits & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
			memTypeIndex = i;
			break;
		}
	}
	if (memTypeIndex == UINT32_MAX) {
		throw std::runtime_error("Failed to find suitable memory type for interop!");
	}
	allocateInfo.memoryTypeIndex = memTypeIndex;

	// Allocate device memory
	m_interopImageMemory = vk::raii::DeviceMemory(m_vkDevice, allocateInfo);
	m_interopImage.bindMemory(*m_interopImageMemory, 0);

	// Get win32 handle from allocated memory
	vk::MemoryGetWin32HandleInfoKHR getHandleInfo{};
	getHandleInfo.memory = *m_interopImageMemory;
	getHandleInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

	auto fpGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
		m_vkInstance.getProcAddr("vkGetMemoryWin32HandleKHR")
	);

	if (!fpGetMemoryWin32HandleKHR) {
		throw std::runtime_error("Failed to load Win32 handle function pointer!");
	}

	if (fpGetMemoryWin32HandleKHR(*m_vkDevice, reinterpret_cast<VkMemoryGetWin32HandleInfoKHR*>(&getHandleInfo), &m_interopImageHandle) != VK_SUCCESS) {
		throw std::runtime_error("Failed to export Win32 memory handle!");
	}

	// Create image view
	vk::ImageViewCreateInfo interopViewInfo{};
	interopViewInfo.image = *m_interopImage;
	interopViewInfo.viewType = vk::ImageViewType::e2D;
	interopViewInfo.format = m_vkSurfaceFormat.format;
	interopViewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	interopViewInfo.subresourceRange.levelCount = 1;
	interopViewInfo.subresourceRange.layerCount = 1;

	m_interopImageView = vk::raii::ImageView(m_vkDevice, interopViewInfo);
	
	m_renderer.resize(m_vkDevice, m_interopImageHandle, m_vkSwapchainExtent, memoryRequirements.size);
}

void Application::InitImGui() {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(m_window.GetGLFWwindow(), true);

	vk::DescriptorPoolSize poolSizes[] = {
		{ vk::DescriptorType::eCombinedImageSampler, 100 },
		{ vk::DescriptorType::eSampler, 100 },
		{ vk::DescriptorType::eSampledImage, 100 }
	};

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	poolInfo.maxSets = 300;
	poolInfo.poolSizeCount = 3;
	poolInfo.pPoolSizes = poolSizes;

	m_imguiDescriptorPool = vk::raii::DescriptorPool(m_vkDevice, poolInfo);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = *m_vkInstance;
	init_info.PhysicalDevice = *m_vkPhysicalDevice;
	init_info.Device = *m_vkDevice;
	init_info.QueueFamily = m_graphicsQueueFamilyIndex;
	init_info.Queue = *m_vkGraphicsQueue;
	init_info.DescriptorPool = *m_imguiDescriptorPool;
	init_info.MinImageCount = 2;
	init_info.ImageCount = static_cast<uint32_t>(m_vkSwapchainImageViews.size());
	init_info.ApiVersion = VK_API_VERSION_1_3;
	init_info.UseDynamicRendering = VK_TRUE;
	init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pNext = nullptr;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;

	VkFormat rawFormat = static_cast<VkFormat>(m_vkSurfaceFormat.format);
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &rawFormat;

	ImGui_ImplVulkan_Init(&init_info);
}

void Application::recreateSwapchain()
{
	int width = 0;
	int height = 0;
	while (width == 0 || height == 0) {
		m_window.getFramebufferSize(&width, &height);

		// Sleep the thread until glfw is resized
		if (width == 0 || height == 0) {
			glfwWaitEvents();
		}
	}

	// Wait for GPU to finish before continuing
	m_vkDevice.waitIdle();

	// Clean up cuda interop objects
	m_interopImageView = nullptr;
	m_interopImageMemory = nullptr;
	m_interopImage = nullptr;

	// Get new surface capabilities
	vk::SurfaceCapabilitiesKHR capabilities = m_vkPhysicalDevice.getSurfaceCapabilitiesKHR(*m_vkSurface);
	m_vkSwapchainExtent = capabilities.currentExtent;

	// Rebuild swapchain
	vk::SwapchainCreateInfoKHR swapchainCreateInfo{};
	swapchainCreateInfo.surface = *m_vkSurface;
	swapchainCreateInfo.minImageCount = static_cast<uint32_t>(m_vkSwapchainImageViews.size());
	swapchainCreateInfo.imageFormat = m_vkSurfaceFormat.format;
	swapchainCreateInfo.imageColorSpace = m_vkSurfaceFormat.colorSpace;
	swapchainCreateInfo.imageExtent = m_vkSwapchainExtent;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
	swapchainCreateInfo.preTransform = capabilities.currentTransform;
	swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainCreateInfo.presentMode = vk::PresentModeKHR::eFifo;
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.oldSwapchain = *m_vkSwapchain;

	vk::raii::SwapchainKHR tempSwapchain(m_vkDevice, swapchainCreateInfo);
	m_vkSwapchain = std::move(tempSwapchain);

	// Grab the swapchain image views
	std::vector<vk::Image> swapchainImages = m_vkSwapchain.getImages();
	m_vkSwapchainImageViews.clear();
	for (const auto& image : swapchainImages) {
		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.image = image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = m_vkSurfaceFormat.format;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;
		m_vkSwapchainImageViews.emplace_back(m_vkDevice, viewInfo);
	}

	InitCUDAVulkanInterop();
}

void Application::renderImGui()
{
	float currentFrameTimeMs = m_frameTimeHistory[m_historyOffset];
	float currentFps = 1000.f / currentFrameTimeMs;

	ImGui::Begin("Test Window");
	if(ImGui::Button("Load Scene File")) {
		m_bOpenLoadSceneModal = true;
	}
	if (ImGui::Button("Load Environment Map")) {
		pickEnvironmentMap();
	}
	if (ImGui::Button("Clear Environment Map")) {
		clearEnvironmentMap();
	}
	if (m_currentEnvironmentMap != nullptr) {
		if (ImGui::SliderFloat("Environment Intensity", &m_currentEnvironmentMap->m_intensity, 0.0f, 10.0f)) {
			m_camera.m_hasChanged = true;
		}
	}
	if (ImGui::Button("Save Render to File")) {
		saveCurrentRender();
	}
	m_renderer.drawRenderSettingsImGui(m_camera);
	ImGui::Text("Performance Stats:");
	ImGui::Text("FPS: %.1f", currentFps);
	ImGui::Text("Frame Time: %.2f ms", currentFrameTimeMs);
	ImGui::PlotLines("##FrameTimeGraph",
		m_frameTimeHistory.data(),
		static_cast<int>(m_frameTimeHistory.size()),
		static_cast<int>(m_historyOffset),
		"Frame Times (ms)",
		0.0f, 33.f,
		ImVec2(0, 80));
	// TODO: Don't query this every frame
	size_t freeBytes = 0;
	size_t totalBytes = 0;
	cudaMemGetInfo(&freeBytes, &totalBytes);
	size_t usedBytes = totalBytes - freeBytes;
	double usedMB = usedBytes / (1024.0 * 1024.0);
	ImGui::Text("Used CUDA Memory: %.2f MB", usedMB);
	ImGui::End();

	drawLoadSceneModal();
}

void Application::pickSceneFile()
{
	NFD::Guard nfdGuard;
	nfdfilteritem_t filterItem[1] = {
		{ "Scene Files",
		"json,gltf,glb,obj" }
	};
	NFD::UniquePath outPath;

	nfdresult_t result = NFD::OpenDialog(outPath, filterItem, 1, "Scene Files");

	if (result != NFD_OKAY) {
		return;
	}

	m_pendingScenePath = outPath.get();
}

void Application::drawLoadSceneModal()
{
	const char* modalName = "Load Scene";

	if (m_bOpenLoadSceneModal) {
		ImGui::OpenPopup(modalName);
		m_bOpenLoadSceneModal = false;
	}

	// Put modal in the center of the screen
	ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (ImGui::BeginPopupModal(modalName, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
		// Pick filepath
		ImGui::SetNextItemWidth(400.0f);
		ImGui::InputText("##ScenePath", m_pendingScenePath.data(), m_pendingScenePath.size() + 1, ImGuiInputTextFlags_ReadOnly);
		ImGui::SameLine();
		if (ImGui::Button("Browse...")) {
			pickSceneFile();
		}

		// Pick up axis if thre is a scene selected
		bool bHasScene = !m_pendingScenePath.empty();

		const std::vector<UpAxis> upAxisOptions = { UpAxis::XUp, UpAxis::YUp, UpAxis::ZUp };
		const std::vector<const char*> upAxisLabels = { "X Up", "Y Up (Default)", "Z Up" };

		int selectedOption = 0;
		for (int i = 0; i < 3; i++) {
			if (upAxisOptions[i] == m_sceneLoadOptions.upAxis) {
				selectedOption = i;
			}
		}

		if (ImGui::BeginCombo("Up Axis", upAxisLabels[selectedOption])) {
			for (int i = 0; i < 3; i++) {
				bool bSelected = (i == selectedOption);
				if (ImGui::Selectable(upAxisLabels[i], bSelected)) {
					m_sceneLoadOptions.upAxis = upAxisOptions[i];
				}
				if (bSelected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}

		ImGui::InputFloat("Import Scale", &m_sceneLoadOptions.importScale, 0.0f, 0.0f, "%g");
		m_sceneLoadOptions.importScale = glm::max(m_sceneLoadOptions.importScale, 1e-6f);

		drawFallbackMaterialImGui();

		ImGui::BeginDisabled(!bHasScene);
		if (ImGui::Button("Load")) {
			m_currentScene = SceneLoader::loadFromFile(m_pendingScenePath, m_sceneLoadOptions);
			m_camera.m_hasChanged = true;
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndDisabled();
		ImGui::SameLine();
		if (ImGui::Button("Cancel")) {
			ImGui::CloseCurrentPopup();
		}

		ImGui::EndPopup();
	}
}


void Application::drawFallbackMaterialImGui()
{
	FallbackMaterialSettings& fallback = m_sceneLoadOptions.fallbackMaterial;
	using Type = FallbackMaterialSettings::Type;

	ImGui::SeparatorText("Fallback Material");
	ImGui::TextDisabled("Used for objects without a material");

	const char* typeLabels[] = { "Diffuse", "Mirror", "Glass", "Glossy", "PBR" };
	int type = static_cast<int>(fallback.type);
	if (ImGui::Combo("Type", &type, typeLabels, IM_ARRAYSIZE(typeLabels))) {
		fallback.type = static_cast<Type>(type);
	}

	ImGui::ColorEdit3("Albedo", &fallback.albedo.x);

	// Only show the settings the selected type uses
	if (fallback.type == Type::Glossy || fallback.type == Type::Pbr) {
		ImGui::SliderFloat("Roughness", &fallback.roughness, 0.0f, 1.0f);
	}
	if (fallback.type == Type::Pbr) {
		ImGui::SliderFloat("Metallic", &fallback.metallic, 0.0f, 1.0f);
	}
	if (fallback.type == Type::Glass) {
		ImGui::SliderFloat("IOR", &fallback.ior, 1.0f, 2.5f);
	}

	ImGui::Checkbox("Override All Materials (Keep Lights)", &m_sceneLoadOptions.bOverrideMaterials);
}

void Application::saveCurrentRender()
{
	NFD::Guard nfdGuard;
	nfdfilteritem_t filterItem[1] = {
		{ "PNG Image",
		"png" }
	};
	NFD::UniquePath outPath;

	nfdresult_t result = NFD::SaveDialog(outPath, filterItem, 1, "render.png");

	if (result != NFD_OKAY) {
		return;
	}

	std::string saveFilePath = outPath.get();

	m_renderer.saveCurrentRenderToFile(saveFilePath);

}

void Application::pickEnvironmentMap()
{
	NFD::Guard nfdGuard;
	nfdfilteritem_t filterItem[1] = {
		{ "Environment Maps",
		"hdr" }
	};
	NFD::UniquePath outPath;

	nfdresult_t result = NFD::OpenDialog(outPath, filterItem, 1, "");

	if (result != NFD_OKAY) {
		return;
	}

	std::string environmentMapPath = outPath.get();

	m_currentEnvironmentMap = std::make_unique<EnvironmentMap>(environmentMapPath);
	m_camera.m_hasChanged = true;
}

void Application::clearEnvironmentMap()
{
	m_currentEnvironmentMap = nullptr;
	m_camera.m_hasChanged = true;
}
