#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <set>
#include <stdexcept>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void saveImage();

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaError_t result = cudaGLSetGLDevice(0);
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA/OpenGL setup failed: %s. Try the --headless option.\n",
            cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaError_t result = cudaGLRegisterBufferObject(pbo);
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA/OpenGL buffer sharing failed: %s. Try the --headless option.\n",
            cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    printf("Opengl Renderer:%s\n", glGetString(GL_RENDERER));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    bool settingsChanged = ImGui::Checkbox("Sort paths by material", &imguiData->MaterialSortingEnabled);
    settingsChanged |= ImGui::Checkbox("Direct area lighting", &imguiData->DirectLightingEnabled);
    settingsChanged |= ImGui::Checkbox("Russian roulette", &imguiData->RussianRouletteEnabled);
    settingsChanged |= ImGui::Checkbox("Compact terminated paths", &imguiData->CompactionEnabled);
    settingsChanged |= ImGui::Checkbox("Mesh BVH", &imguiData->MeshBVHEnabled);
    settingsChanged |= ImGui::Checkbox("Stochastic antialiasing", &imguiData->AntialiasingEnabled);
    if (settingsChanged) camchanged = true;
    ImGui::Text("GPU trace %.2f ms/iteration", imguiData->LastTraceMs);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json [--headless] [--sort on|off] [--compact on|off]\n"
            "  [--direct-light on|off] [--rr on|off] [--bvh on|off] [--aa on|off]\n"
            "  [--iterations N] [--depth N] [--aperture R] [--seed N]\n"
            "  [--output PREFIX] [--timings FILE.csv] [--stats FILE.csv]\n"
            "  [--checkpoints 1,8,32,128] [--no-save]\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];
    bool headless = false;
    bool noSave = false;
    std::string timingsPath, statisticsPath;
    std::set<int> checkpoints;

    // Load scene file
    try { scene = new Scene(sceneFile); }
    catch (const std::exception& error)
    {
        fprintf(stderr, "Scene error: %s\n", error.what());
        return 1;
    }

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();
    try
    {
        for (int i = 2; i < argc; ++i)
        {
            std::string option = argv[i];
            if (option == "--headless") { headless = true; continue; }
            if (option == "--no-save") { noSave = true; continue; }
            if (++i >= argc) throw std::runtime_error("Missing value for " + option);
            std::string value = argv[i];
            auto switchValue = [&]() {
                if (value != "on" && value != "off") throw std::runtime_error(option + " expects on or off");
                return value == "on";
            };
            if (option == "--sort") guiData->MaterialSortingEnabled = switchValue();
            else if (option == "--compact") guiData->CompactionEnabled = switchValue();
            else if (option == "--direct-light") guiData->DirectLightingEnabled = switchValue();
            else if (option == "--rr") guiData->RussianRouletteEnabled = switchValue();
            else if (option == "--bvh") guiData->MeshBVHEnabled = switchValue();
            else if (option == "--aa") guiData->AntialiasingEnabled = switchValue();
            else if (option == "--iterations")
            {
                int count = std::stoi(value);
                if (count < 1 || count > 1000000) throw std::runtime_error("Iterations must be 1..1000000");
                scene->state.iterations = count;
            }
            else if (option == "--depth")
            {
                int depth = std::stoi(value);
                if (depth < 1 || depth > 64) throw std::runtime_error("Depth must be 1..64");
                scene->state.traceDepth = depth;
            }
            else if (option == "--seed")
            {
                guiData->SeedOffset = std::stoi(value);
                if (guiData->SeedOffset < 0 || guiData->SeedOffset > 1000000)
                    throw std::runtime_error("Seed offset must be 0..1000000");
            }
            else if (option == "--aperture")
            {
                scene->state.camera.aperture = std::stof(value);
                if (!std::isfinite(scene->state.camera.aperture) || scene->state.camera.aperture < 0)
                    throw std::runtime_error("Aperture must be finite and nonnegative");
            }
            else if (option == "--output") scene->state.imageName = value;
            else if (option == "--timings") timingsPath = value;
            else if (option == "--stats") statisticsPath = value;
            else if (option == "--checkpoints")
            {
                std::istringstream values(value);
                std::string item;
                while (std::getline(values, item, ',')) checkpoints.insert(std::stoi(item));
            }
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (!headless && (!timingsPath.empty() || !statisticsPath.empty() || noSave || !checkpoints.empty()))
            throw std::runtime_error("Timing, statistics, checkpoints and no-save options require --headless");
    }
    catch (const std::exception& error)
    {
        fprintf(stderr, "Argument error: %s\n", error.what());
        return 1;
    }

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    if (headless)
    {
        auto openCSV = [](const std::string& filename, std::ofstream& stream) {
            if (filename.empty()) return;
            std::filesystem::path path(filename);
            if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
            stream.open(path);
            if (!stream) throw std::runtime_error("Cannot write CSV: " + filename);
            stream << std::setprecision(9);
        };
        std::ofstream timings, statistics;
        try
        {
            openCSV(timingsPath, timings);
            openCSV(statisticsPath, statistics);
            std::filesystem::path output(renderState->imageName);
            if (!noSave && output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
        }
        catch (const std::exception& error)
        {
            fprintf(stderr, "Output error: %s\n", error.what());
            return 1;
        }
        if (timings) timings << "iteration,gpu_ms\n";
        if (statistics) statistics << "iteration,bounce,rays_in,rays_alive,rays_scheduled,camera_ms,intersection_ms,sorting_ms,shading_ms,gather_ms,compaction_ms,iteration_gpu_ms\n";
        enableTraceStatistics(!statisticsPath.empty());
        InitDataContainer(guiData);
        pathtraceInit(scene);
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        printf("GPU: %s | %d x %d | depth %d | sort %d compact %d direct %d rr %d bvh %d aa %d\n",
            properties.name, width, height, renderState->traceDepth,
            guiData->MaterialSortingEnabled, guiData->CompactionEnabled,
            guiData->DirectLightingEnabled, guiData->RussianRouletteEnabled,
            guiData->MeshBVHEnabled, guiData->AntialiasingEnabled);
        auto renderStart = std::chrono::steady_clock::now();
        for (iteration = 1; iteration <= static_cast<int>(renderState->iterations); ++iteration)
        {
            pathtrace(NULL, 0, iteration);
            if (timings) timings << iteration << ',' << guiData->LastTraceMs << '\n';
            if (statistics)
            {
                for (const auto& s : lastBounceStatistics())
                    statistics << iteration << ',' << s.depth << ',' << s.raysIn << ','
                        << s.raysAlive << ',' << s.raysScheduled << ',' << s.cameraMs << ','
                        << s.intersectionMs << ',' << s.sortingMs << ',' << s.shadingMs << ','
                        << s.gatherMs << ',' << s.compactionMs << ',' << guiData->LastTraceMs << '\n';
            }
            if (!noSave && checkpoints.count(iteration) && iteration < static_cast<int>(renderState->iterations)) saveImage();
            if (iteration == 1 || iteration % 8 == 0 ||
                iteration == static_cast<int>(renderState->iterations))
            {
                printf("Rendered %d / %u samples\n", iteration,
                    renderState->iterations);
            }
        }
        printf("Render loop wall time: %.3f s (initialization and final image encoding excluded)\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - renderStart).count());
        iteration = static_cast<int>(renderState->iterations);
        if (!noSave) saveImage();
        pathtraceFree();
        cudaDeviceReset();
        return 0;
    }

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    ogLookAt = cam.lookAt;
    glm::vec3 cameraOffset = cam.position - ogLookAt;
    zoom = glm::length(cameraOffset);
    phi = std::atan2(cameraOffset.x, cameraOffset.z);
    theta = std::acos(glm::clamp(cameraOffset.y / zoom, -1.0f, 1.0f));

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    float samples = std::max(1, iteration);
    // output image file
    Image img(width, height);
    Image hdr(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = glm::max(renderState->image[index] / samples,
                glm::vec3(0.0f));
            hdr.setPixel(width - 1 - x, y, pix);
            glm::vec3 mapped = pix / (glm::vec3(1.0f) + pix);
            img.setPixel(width - 1 - x, y,
                glm::pow(mapped, glm::vec3(1.0f / 2.2f)));
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    hdr.saveHDR(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::normalize(glm::cross(v, u));
        cam.up = glm::normalize(glm::cross(r, v));
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                saveImage();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
