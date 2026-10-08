#include "window.h"

#include "application.h"

#include <stdexcept>

Window::Window(int width, int height, const char* title) {
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_glfwWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);

	if (!m_glfwWindow) {
		throw std::runtime_error("Failed to create GLFW window");
	}

	glfwSetFramebufferSizeCallback(m_glfwWindow, FramebufferResizeCallback);
}

Window::Window(Window&& other) noexcept : 
	m_glfwWindow(other.m_glfwWindow) 
{
	other.m_glfwWindow = nullptr;
}

Window& Window::operator=(Window&& other) noexcept {
	if (this != &other) {
		if (m_glfwWindow) {
			glfwDestroyWindow(m_glfwWindow);
		}
		m_glfwWindow = other.m_glfwWindow;
		other.m_glfwWindow = nullptr;
	}
	return *this;
}

void Window::getFramebufferSize(int* width, int* height)
{
	glfwGetFramebufferSize(m_glfwWindow, width, height);
}

void Window::FramebufferResizeCallback(GLFWwindow* glfwWindow, int width, int height)
{
	auto application = reinterpret_cast<Application*>(glfwGetWindowUserPointer(glfwWindow));
	if (application) {
		application->getWindow().m_hasBeenResized = true;
	}
}

Window::~Window() {
	if (m_glfwWindow) {
		glfwDestroyWindow(m_glfwWindow);
	}
}