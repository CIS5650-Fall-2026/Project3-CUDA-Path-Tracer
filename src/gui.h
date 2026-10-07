#pragma once

// forward declare

struct GLFWwindow;
struct Camera;
class Scene;
class GuiDataContainer;

struct GuiRefs {
    Camera* camera;
    const Camera* og_camera;
    Scene* scene;
    GuiDataContainer* data;
    int iteration;
};

namespace gui {
void init(GLFWwindow* window);
void begin_frame(bool &mouse_over_imgui_window);
bool render_imgui(GuiRefs& refs);
void end_frame();
void shutdown();
}


