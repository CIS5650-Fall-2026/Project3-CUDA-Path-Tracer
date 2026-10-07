#include "camera.h"
#include "glsl_utility.hpp"
#include "gui.h"
#include "gui_data.h"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "utilities.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>

#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

static std::string start_time_string;

// For camera controls
static double last_x;
static double last_y;
static double last_time;
static constexpr float MAX_FRAME_DT = 0.12f;

static bool needs_reset = true;

Camera camera;
Camera og_camera;
CameraController camera_controller;

Scene* scene;
GuiDataContainer* gui_data;
RenderState* render_state;
int iteration;

int width;
int height;

GLuint position_location = 0;
GLuint texcoords_location = 1;
GLuint display_image;
GLuint pbo;
cudaGraphicsResource_t cuda_pixel_resource;

GLFWwindow* window;
GuiDataContainer* imgui_data = NULL;
bool mouse_over_imgui_window = false;

// Forward declarations for window loop and interactivity
void run_cuda();
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_position_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

std::string current_time_string() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init_textures() {
    glGenTextures(1, &display_image);
    glBindTexture(GL_TEXTURE_2D, display_image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void init_vao(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    };

    GLfloat texcoords[] = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    GLushort indices[] = {0, 1, 3, 3, 1, 2};

    GLuint vertex_buffer_obj_id[3];
    glGenBuffers(3, vertex_buffer_obj_id);

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_id[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)position_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_location);

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_id[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoords_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoords_location);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_buffer_obj_id[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint init_shader() {
    const char* attrib_locations[] = {"Position", "Texcoords"};
    GLuint program = glsl_utility::create_default_program(attrib_locations, 2);
    GLint location;

    // glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void delete_pbo(GLuint* pbo) {
    if (pbo) {
        if (cuda_pixel_resource) {
            cudaGraphicsUnregisterResource(cuda_pixel_resource);
        }
        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void delete_texture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanup_cuda() {
    if (pbo) {
        delete_pbo(&pbo);
    }
    if (display_image) {
        delete_texture(&display_image);
    }
}

void init_cuda() {
    cudaSetDevice(0);
}

void init_pbo() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int texture_data_size = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, texture_data_size, NULL, GL_DYNAMIC_COPY);
    cudaGraphicsGLRegisterBuffer(&cuda_pixel_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void error_callback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));

    // Set up ImGui
    gui::init(window);

    // Initialize other stuff
    init_vao();
    init_textures();
    init_cuda();
    init_pbo();
    GLuint passthrough_program = init_shader();

    glUseProgram(passthrough_program);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void init_imgui_data(GuiDataContainer* gui_data) {
    imgui_data = gui_data;
}

bool is_mouse_over_imgui_window() {
    return mouse_over_imgui_window;
}

void save_image(bool on_exit) {
    copy_image_to_host();

    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = render_state->image[index];
            img.set_pixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = render_state->image_name;
    std::ostringstream ss;
    ss << filename << "." << start_time_string << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    std::string dir = (on_exit ? "renders" : "saves") + std::string("/");
    img.save_png(dir + filename, scene->state.settings.agx);
    // img.save_hdr(filename);  // Save a Radiance HDR file
}

void save_scene(bool on_exit) {
    std::string dir = on_exit ? "scene_renders/" : "scene_saves/";
    std::string time = on_exit ? start_time_string : current_time_string();
    std::string path = dir + render_state->image_name + "." + time + ".json";
    render_state->controls = camera.settings;
    scene->save_to_json(path);
    printf("Saved %s\n", path.c_str());
}

std::string open_scene_dialog() {
    std::filesystem::create_directories("scene_saves");
    std::string initial_dir = std::filesystem::absolute("scene_saves").string();
    char file[MAX_PATH] = "";
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "Scene (*.json)\0*.json\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrInitialDir = initial_dir.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn) ? file : "";
}

void init_camera_from_scene() {
    iteration = 0;
    render_state = &scene->state;
    const CameraData& cam = render_state->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    camera.position = cam.position;
    camera.yaw = glm::atan(-cam.view.x, -cam.view.z);
    camera.pitch = glm::asin(cam.view.y);
    camera.settings = render_state->controls;
    og_camera = camera;
}

void load_scene(const std::string& path) {
    pathtrace_free();
    delete scene;
    scene = new Scene(path);

    int old_width = width, old_height = height;
    init_camera_from_scene();
    if (width != old_width || height != old_height) {
        cleanup_cuda();
        glfwSetWindowSize(window, width, height);
        glViewport(0, 0, width, height);
        init_textures();
        init_pbo();
    }

    pathtrace_init(scene);
    needs_reset = true;
}

void main_loop() {
    pathtrace_init(scene);
    pathtrace_reset(scene);

    last_time = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt = glm::min((float)(now - last_time), MAX_FRAME_DT);
        last_time = now;

        gui::begin_frame(mouse_over_imgui_window);
        GuiRefs refs{&camera, &og_camera, scene, gui_data, iteration};
        needs_reset |= gui::render_imgui(refs);
        needs_reset |= camera_controller.update_camera(camera, dt);

        run_cuda();

        std::string title = "CIS565 Path Tracer | " +
                            utility_core::convert_int_to_string(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, display_image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        gui::end_frame();
        glfwSwapBuffers(window);

        if (gui_data->env_requested) {
            gui_data->env_requested = false;
            try {
                scene->set_environment(gui_data->env_path);
            } catch (const std::exception& e) {
                std::cerr << e.what() << '\n';
            }
            if (!scene->env.pixels.empty() && render_state->settings.env_strength == 0.f) {
                render_state->settings.env_strength = 1.f;
            }
            pathtrace_upload_env(*scene);
            needs_reset = true;
        }
        if (gui_data->load_requested) {
            gui_data->load_requested = false;
            std::string path = open_scene_dialog();
            if (!path.empty()) {
                load_scene(path);
            }
        }
    }

    save_image(true);
    save_scene(true);

    pathtrace_free();
    cleanup_cuda();
    cudaDeviceReset();

    gui::shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    start_time_string = current_time_string();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* scene_file = argv[1];

    // Load scene file
    scene = new Scene(scene_file);

    // Create ImGui data instance
    gui_data = new GuiDataContainer();

    init_camera_from_scene();

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    init_imgui_data(gui_data);
    init_data_container(gui_data);

    // GLFW main loop
    main_loop();

    return 0;
}

void reset_accumulation() {
    iteration = 0;
    camera.write_data(render_state->camera);
    needs_reset = false;
}

void run_cuda() {
    if (needs_reset) {
        reset_accumulation();
        pathtrace_reset(scene);
    }

    if (iteration < render_state->iterations) {
        cudaGraphicsMapResources(1, &cuda_pixel_resource);

        uchar4* pbo_dptr;
        size_t bytes;
        iteration++;

        cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr, &bytes, cuda_pixel_resource);

        // execute the kernel
        pathtrace(pbo_dptr, iteration);

        // unmap buffer object
        cudaGraphicsUnmapResources(1, &cuda_pixel_resource);
    } else {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (camera_controller.process_keyboard(key, action != GLFW_RELEASE)) {
        return;
    }

    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            if (mods & GLFW_MOD_CONTROL) {
                save_scene(false);
            } else {
                save_image(false);
            }
            break;
        case GLFW_KEY_H:
            camera = og_camera;
            needs_reset = true;
            break;
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_RIGHT ||
        (action == GLFW_PRESS && is_mouse_over_imgui_window())) {
        return;
    }

    bool captured = action == GLFW_PRESS;
    camera_controller.set_captured(captured);
    glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    glfwGetCursorPos(window, &last_x, &last_y);
}

void mouse_position_callback(GLFWwindow* window, double xpos, double ypos) {
    camera_controller.handle_mouse(xpos - last_x, ypos - last_y);
    last_x = xpos;
    last_y = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!is_mouse_over_imgui_window()) {
        camera_controller.handle_mouse_scroll(yoffset);
    }
}
