#pragma once
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


struct CameraData {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 look_at;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixel_length;
    float lens_radius;
    float focal_distance;

    void set_fov(float fovy);
};

struct CameraConfig {
    float accel = 100.f;
    float damping = 5.f;
    float mouse_sens = 0.15f;
    float scroll_sens = 0.1f;
};

struct Camera {
    glm::vec3 position = glm::vec3(0.f);
    glm::vec3 velocity = glm::vec3(0.f);
    float yaw = 0.f;
    float pitch = 0.f;
    float aspect = 1.5f;

    CameraConfig settings;

    Camera() = default;
    explicit Camera(const CameraConfig& config);

    glm::quat orient() const;
    glm::vec3 forward() const;
    glm::vec3 right() const;
    glm::mat4 view_matrix() const;
    void reset_view();
    void write_data(CameraData& data) const;
};

class CameraInput {
  public:
    bool process_keyboard(int key, bool pressed);
    void clear();
    glm::vec3 movement_vector(const Camera& camera) const;

  private:
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
};

class CameraController {
  public:
    void set_captured(bool captured);
    bool is_captured() const;
    bool process_keyboard(int key, bool pressed);
    void handle_mouse(double mouse_dx, double mouse_dy);
    void handle_mouse_scroll(double scroll);
    bool update_camera(Camera& camera, float dt);

  private:
    void reset_frame_input();

    CameraInput input;
    bool captured = false;
    float rot_hor = 0.f;
    float rot_vert = 0.f;
    float scroll = 0.f;
};
