#include "camera.h"
#include "math_utils.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

static constexpr float MOUSE_SENS_SCALE = 0.02f;
static constexpr float SAFE_FRAC_PI_2 = PI * 0.5f - 0.0001f;
static constexpr float MIN_SPEED2 = 1e-6f;

void CameraData::set_fov(float fovy) {
    float yscaled = tan(fovy / 2.f * (PI / 180));
    float xscaled = (yscaled * resolution.x) / resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    fov = glm::vec2(fovx, fovy);
    pixel_length = glm::vec2(2 * xscaled / (float)resolution.x,
                                    2 * yscaled / (float)resolution.y);
}

Camera::Camera(const CameraConfig& config) : settings(config) {}

glm::quat Camera::orient() const {
    glm::quat yaw_q   = glm::angleAxis(yaw,   glm::vec3(0.f, 1.f, 0.f));
    glm::quat pitch_q = glm::angleAxis(pitch, glm::vec3(1.f, 0.f, 0.f));
    return yaw_q * pitch_q;
}

glm::vec3 Camera::forward() const {
    return orient() * glm::vec3(0.f, 0.f, -1.f);
}

glm::vec3 Camera::right() const {
    return orient() * glm::vec3(1.f, 0.f, 0.f);
}

glm::mat4 Camera::view_matrix() const {
    return glm::lookAtRH(position, position + forward(), glm::vec3(0.f, 1.f, 0.f));
}

void Camera::reset_view() {
    *this = Camera(settings);
}

void Camera::write_data(CameraData& data) const {
    data.position = position;
    data.view = forward();
    data.right = right();
    data.up = glm::cross(data.right, data.view);
}

bool CameraInput::process_keyboard(int key, bool pressed) {
    switch (key) {
    case GLFW_KEY_W: forward = pressed; break;
    case GLFW_KEY_S: backward = pressed; break;
    case GLFW_KEY_A: left = pressed; break;
    case GLFW_KEY_D: right = pressed; break;
    case GLFW_KEY_SPACE: up = pressed; break;
    case GLFW_KEY_LEFT_SHIFT: down = pressed; break;
    default: return false;
    }
    return true;
}

void CameraInput::clear() {
    *this = CameraInput();
}

static float axis(bool positive, bool negative) {
    return (float)positive - (float)negative;
}

glm::vec3 CameraInput::movement_vector(const Camera& camera) const {
    glm::vec3 dir = camera.forward() * axis(forward, backward) +
                    camera.right() * axis(right, left) +
                    glm::vec3(0.f, 1.f, 0.f) * axis(up, down);

    return glm::dot(dir, dir) > 0.f ? glm::normalize(dir) : glm::vec3(0.f);
}

void CameraController::set_captured(bool captured) {
    this->captured = captured;

    if (!captured) {
        input.clear();
    }
}

bool CameraController::is_captured() const {
    return captured;
}

bool CameraController::process_keyboard(int key, bool pressed) {
    return is_captured() && input.process_keyboard(key, pressed);
}

void CameraController::handle_mouse(double mouse_dx, double mouse_dy) {
    if (is_captured()) {
        rot_hor += (float)mouse_dx;
        rot_vert += (float)mouse_dy;
    }
}

void CameraController::handle_mouse_scroll(double scroll) {
    this->scroll = -(float)scroll * 100.f;
}

bool CameraController::update_camera(Camera& camera, float dt) {
    glm::vec3 dir(0.f);
    if (is_captured()) {
        camera.settings.accel = glm::clamp(
            camera.settings.accel - scroll * camera.settings.scroll_sens * dt, 0.f, 10000.f);
        dir = input.movement_vector(camera);
    } else {
        camera.position += camera.forward() * -scroll * camera.settings.scroll_sens * dt;
    }

    camera.velocity += dir * camera.settings.accel * dt;
    camera.velocity *= glm::exp(-camera.settings.damping * dt);
    if (glm::dot(camera.velocity, camera.velocity) < MIN_SPEED2) {
        camera.velocity = glm::vec3(0.f);
    }
    camera.position += camera.velocity * dt;

    float mouse_sens = camera.settings.mouse_sens * MOUSE_SENS_SCALE;
    camera.yaw -= rot_hor * mouse_sens;
    camera.pitch = glm::clamp(camera.pitch - rot_vert * mouse_sens, -SAFE_FRAC_PI_2, SAFE_FRAC_PI_2);

    bool moved = camera.velocity != glm::vec3(0.f) || rot_hor != 0.f || rot_vert != 0.f ||
                 scroll != 0.f;
    reset_frame_input();
    return moved;
}

void CameraController::reset_frame_input() {
    rot_hor = 0.f;
    rot_vert = 0.f;
    scroll = 0.f;
}
