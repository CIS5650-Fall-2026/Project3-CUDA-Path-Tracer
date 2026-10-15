#include "image.h"
#include "scene.h"
#include "sceneStructs.h"
#include "sceneUtils.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GLFW/glfw3.h>
#include "imgui.h"

#include "vk/context.h"
#include "vk/renderer.h"
#include "vk/swapchain.h"
#include "vk/gpuscene.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static bool showGui = true;
static bool enableDOF = false;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;

int width;
int height;

std::unique_ptr<VulkanContext> context;
std::unique_ptr<Swapchain> swapchain;
std::unique_ptr<GpuScene> gpuScene;
std::unique_ptr<Renderer> renderer;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;

// Forward declarations for window loop and interactivity
bool initGLFW();
void updateCamera();
void cleanup();
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

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

bool initGLFW()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return false;
    }

    // We're using Vulkan, so avoid creating an OpenGL context 
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    if (!glfwVulkanSupported())
    {
        glfwTerminate();
        fprintf(stderr, "Vulkan not supported by GLFW\n");
        return false;
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        fprintf(stderr, "Failed to create GLFW window\n");
        return false;
    }

    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    return true;
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    if (!initGLFW())
    {
        return false;
    }

    // Initialize VulkanContext
    context = std::make_unique<VulkanContext>();
    
    if (!context->init(window))
    {
        fprintf(stderr, "Failed to initialize VulkanContext\n");
        return false;
    }

    // Initialize Swapchain
    swapchain = std::make_unique<Swapchain>();
    
    if (!swapchain->init(*context, width, height))
    {
        fprintf(stderr, "Failed to initialize Swapchain\n");
        return false;
    }

    // Initialize GPU scene data
    gpuScene = std::make_unique<GpuScene>();
    {
        const sceneutil::FlatScene flatScene = sceneutil::flattenScene(*scene);
        if (!gpuScene->init(*context, flatScene))
        {
            fprintf(stderr, "Failed to initialize GpuScene\n");
            return false;
        }
    }

    // Initialize Renderer
    renderer = std::make_unique<Renderer>();

    if (!renderer->init(*context, *swapchain, window, width, height, *gpuScene))
    {
        fprintf(stderr, "Failed to initialize Renderer\n");
        return false;
    }

    renderer->setMaxTraceDepth(renderState->traceDepth);
    
    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}

// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    Gui& gui = renderer->getGui();
    gui.beginFrame();

    if (showGui)
    {
        ImGui::Begin("Path Tracer Analytics");

        //ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::Checkbox("Enable Depth Of Field", &enableDOF))
        {
            renderer->setDepthOfField(enableDOF);
            renderer->resetRenderedFrameCount();
        }

        ImGui::End();
    }
    gui.renderFrame();
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Sleep as long as the window is minimized
        int curWidth = 0;
        int curHeight = 0;
        glfwGetFramebufferSize(window, &curWidth, &curHeight);
        if (curWidth == 0 || curHeight == 0)
        {
            glfwWaitEvents();
            continue;
        }

        updateCamera();

        std::string title = "CIS565 Path Tracer | " 
            + utilityCore::convertIntToString(static_cast<int>(renderer->getRenderedFrameCount()))
            + " Iterations";
        glfwSetWindowTitle(window, title.c_str());

        // Render ImGui Stuff
        RenderImGui();
        
        if (!renderer->drawFrame())
        {
            fprintf(stderr, "drawFrame() call failed.\n");
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
}

void cleanup()
{
    if (context)
    {
        vkDeviceWaitIdle(context->getDevice());
    }
    
    renderer.reset();
    gpuScene.reset();
    swapchain.reset();
    context.reset();

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
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = glm::normalize(cam.view);
    glm::vec3 up = glm::normalize(cam.up);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize our rendering infrastructure
    if (!init())
    {
        // Initialization failed
        return 1;
    }

    // Initialize ImGui Data
    InitImguiData(guiData);

    // GLFW main loop
    mainLoop();
    cleanup();

    return 0;
}

void updateCamera()
{
    Camera& cam = renderState->camera;

    if (camchanged)
    {
        renderer->resetRenderedFrameCount();
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

    renderer->setCameraParams({
        cam.position,
        cam.view,
        cam.right,
        cam.up,
        cam.pixelLength
    });
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
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_S:
                showGui = !showGui;
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
    if (renderer->getGui().wantsMouse())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX && ypos == lastY)
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
