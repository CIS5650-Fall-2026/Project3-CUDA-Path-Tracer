#pragma once

#include "glfw/glfw3.h"

#include <vulkan/vulkan.h>

class Window {
public:
	Window() = default;
	Window(int width, int height, const char* title);
	~Window();

	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;

	Window(Window&& other) noexcept;
	Window& operator=(Window&& other) noexcept;

	GLFWwindow* GetGLFWwindow() const { return m_glfwWindow; }

	void getFramebufferSize(int* width, int* height);

	bool m_hasBeenResized = false;

private:
	static void FramebufferResizeCallback(GLFWwindow* glfwWindow, int width, int height);

	GLFWwindow* m_glfwWindow = nullptr;
};