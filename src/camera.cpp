#include "camera.h"

#include <glfw/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include "mathHelpers.h"

Camera::Camera(glm::vec3 eye, glm::vec3 lookAt, glm::vec3 up, float fovy)
{
	m_position = eye;
	m_look = glm::normalize(lookAt - eye);
	m_up = glm::normalize(up);
	m_right = glm::normalize(glm::cross(m_look, m_up));
	m_fovy = fovy;
}

void Camera::tick(float deltaTime, const InputState& inputState)
{
	glm::vec3 moveDirection = glm::vec3(0.0f);
	if (inputState.isKeyPressed(GLFW_KEY_W)) {
		moveDirection += m_look;
	}
	if (inputState.isKeyPressed(GLFW_KEY_S)) {
		moveDirection -= m_look;
	}
	if (inputState.isKeyPressed(GLFW_KEY_D)) {
		moveDirection += m_right;
	}
	if (inputState.isKeyPressed(GLFW_KEY_A)) {
		moveDirection -= m_right;
	}
	if (inputState.isKeyPressed(GLFW_KEY_SPACE)) {
		moveDirection += m_up;
	}
	if (inputState.isKeyPressed(GLFW_KEY_LEFT_SHIFT)) {
		moveDirection -= m_up;
	}

	moveDirection = MathHelpers::safeNormalize(moveDirection);

	if (moveDirection != glm::vec3(0.0f)) {
		m_position += m_moveSpeed * deltaTime * moveDirection;
		m_hasChanged = true;
	}
}

void Camera::rotate(const glm::vec2& mouseDelta)
{
	if (mouseDelta.x == 0.0f && mouseDelta.y == 0.0f) {
		return;
	}

	float yawAngle = -mouseDelta.x * m_rotateSpeed;
	float pitchAngle = -mouseDelta.y * m_rotateSpeed;

	const glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);

	// Rotate look horizontally (yaw)
	glm::mat4 yawRotateMatrix = glm::rotate(glm::mat4(1.0f), yawAngle, globalUp);
	m_look = glm::normalize(glm::vec3(yawRotateMatrix * glm::vec4(m_look, 0.0f)));

	// Rotate look vertically (pitch)
	glm::mat4 pitchRotateMatrix = glm::rotate(glm::mat4(1.0f), pitchAngle, m_right);
	m_look = glm::normalize(glm::vec3(pitchRotateMatrix * glm::vec4(m_look, 0.0f)));

	m_right = glm::normalize(glm::cross(m_look, globalUp));
	m_up = glm::normalize(glm::cross(m_right, m_look));

	m_hasChanged = true;
}

void Camera::setScrollOffset(float offset)
{
	m_moveSpeed = std::max(0.1f, m_moveSpeed - offset * 0.2f);
}

glm::mat4 Camera::getViewProjectionMatrix(float width, float height) const
{
	glm::mat4 view = glm::lookAt(m_position, m_position + m_look, m_up);

	glm::mat4 projection = glm::perspective(glm::radians(m_fovy), width / height, 0.1f, 1000.f);

	return projection * view;
}
