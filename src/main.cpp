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

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

static std::string startTimeString;
static std::uint64_t saveSequence = 0;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;
static bool cursorPositionValid = false;

static bool camchanged = false;

float zoom, theta, phi;
float originalZoom, originalTheta, originalPhi;
glm::vec3 cameraPosition;
Camera originalCamera;

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;
bool pathTracerInitialized = false;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

namespace
{

struct PreviewOptions
{
    std::string sceneFile;
    std::optional<bool> sort;
    std::optional<bool> wavefront;
    std::optional<bool> rr;
    std::optional<int> rrStart;
    std::optional<RendererSampler> sampler;
    std::optional<bool> motion;
    bool help = false;
};

std::string previewUsage(std::string_view program)
{
    std::ostringstream text;
    text << "Usage: " << program << " SCENEFILE.json [options]\n"
         << "  --sort on|off\n"
         << "  --wavefront on|off\n"
         << "  --rr on|off\n"
         << "  --rr-start N             Completed scattering events before roulette (1..4096)\n"
         << "  --sampler counter|halton Sampling algorithm (default: counter)\n"
         << "  --motion on|off          Rigid-translation motion blur (default: on)\n"
         << "  --help\n";
    return text.str();
}

bool parsePreviewSwitch(std::string_view text, std::string_view option)
{
    if (text == "on") return true;
    if (text == "off") return false;
    throw std::invalid_argument(std::string(option)
        + " expects exactly 'on' or 'off'");
}

RendererSampler parsePreviewSampler(std::string_view text)
{
    if (text == "counter") return RendererSampler::Counter;
    if (text == "halton") return RendererSampler::Halton;
    throw std::invalid_argument(
        "--sampler expects exactly 'counter' or 'halton'");
}

std::uint64_t parsePreviewUnsigned(std::string_view text,
                                   std::string_view option)
{
    if (text.empty() || text.front() == '+' || text.front() == '-')
    {
        throw std::invalid_argument(std::string(option)
            + " expects an unsigned integer");
    }
    std::uint64_t result = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto conversion = std::from_chars(begin, end, result, 10);
    if (conversion.ec == std::errc::result_out_of_range)
    {
        throw std::out_of_range(std::string(option) + " is out of range");
    }
    if (conversion.ec != std::errc() || conversion.ptr != end)
    {
        throw std::invalid_argument(std::string(option)
            + " expects an unsigned integer");
    }
    return result;
}

PreviewOptions parsePreviewOptions(int argc, char** argv)
{
    PreviewOptions options;
    bool positionalOnly = false;
    const auto requireValue = [&](int& index,
                                  std::string_view option) -> std::string_view
    {
        if (index + 1 >= argc)
        {
            throw std::invalid_argument(std::string(option) + " requires a value");
        }
        return argv[++index];
    };
    const auto rejectDuplicate = [](bool duplicate, std::string_view option)
    {
        if (duplicate)
        {
            throw std::invalid_argument(std::string(option)
                + " was specified more than once");
        }
    };

    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (!positionalOnly && argument == "--")
        {
            positionalOnly = true;
        }
        else if (!positionalOnly && argument == "--help")
        {
            rejectDuplicate(options.help, argument);
            options.help = true;
        }
        else if (!positionalOnly && argument == "--wavefront")
        {
            rejectDuplicate(options.wavefront.has_value(), argument);
            options.wavefront = parsePreviewSwitch(
                requireValue(index, argument), argument);
        }
        else if (!positionalOnly && argument == "--sort")
        {
            rejectDuplicate(options.sort.has_value(), argument);
            options.sort = parsePreviewSwitch(
                requireValue(index, argument), argument);
        }
        else if (!positionalOnly && argument == "--rr")
        {
            rejectDuplicate(options.rr.has_value(), argument);
            options.rr = parsePreviewSwitch(
                requireValue(index, argument), argument);
        }
        else if (!positionalOnly && argument == "--rr-start")
        {
            rejectDuplicate(options.rrStart.has_value(), argument);
            const std::uint64_t value = parsePreviewUnsigned(
                requireValue(index, argument), argument);
            if (value == 0 || value > 4096)
            {
                throw std::out_of_range("--rr-start must be in [1, 4096]");
            }
            options.rrStart = static_cast<int>(value);
        }
        else if (!positionalOnly && argument == "--sampler")
        {
            rejectDuplicate(options.sampler.has_value(), argument);
            options.sampler = parsePreviewSampler(
                requireValue(index, argument));
        }
        else if (!positionalOnly && argument == "--motion")
        {
            rejectDuplicate(options.motion.has_value(), argument);
            options.motion = parsePreviewSwitch(
                requireValue(index, argument), argument);
        }
        else if (!positionalOnly && argument.size() >= 2
                 && argument[0] == '-' && argument[1] == '-')
        {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
        else
        {
            if (!options.sceneFile.empty())
            {
                throw std::invalid_argument("more than one scene path was supplied");
            }
            options.sceneFile = argument;
        }
    }

    if (!options.help && options.sceneFile.empty())
    {
        throw std::invalid_argument("a scene JSON path is required");
    }
    return options;
}

} // namespace

// Forward declarations for window loop and interactivity
void runCuda();
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

glm::vec3 normalizedOr(const glm::vec3& value, const glm::vec3& fallback)
{
    const float squaredLength = glm::dot(value, value);
    if (!(squaredLength > 1.0e-12f) || !std::isfinite(squaredLength))
    {
        return fallback;
    }
    const glm::vec3 result = value / std::sqrt(squaredLength);
    return std::isfinite(result.x) && std::isfinite(result.y)
            && std::isfinite(result.z) ? result : fallback;
}

void updateOrbitCamera(Camera& camera)
{
    constexpr float poleEpsilon = 1.0e-4f;
    zoom = std::max(zoom, 0.1f);
    theta = std::clamp(theta, poleEpsilon,
        static_cast<float>(PI) - poleEpsilon);

    const glm::vec3 offset(
        zoom * std::sin(phi) * std::sin(theta),
        zoom * std::cos(theta),
        zoom * std::cos(phi) * std::sin(theta));
    camera.position = camera.lookAt + offset;
    camera.view = normalizedOr(camera.lookAt - camera.position,
        normalizedOr(originalCamera.view, glm::vec3(0.0f, 0.0f, -1.0f)));

    const glm::vec3 referenceUp = normalizedOr(originalCamera.up,
        glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 rightCandidate = glm::cross(camera.view, referenceUp);
    if (glm::dot(rightCandidate, rightCandidate) <= 1.0e-12f)
    {
        rightCandidate = camera.right
            - camera.view * glm::dot(camera.right, camera.view);
    }
    if (glm::dot(rightCandidate, rightCandidate) <= 1.0e-12f)
    {
        const glm::vec3 helper = std::fabs(camera.view.y) < 0.999f
            ? glm::vec3(0.0f, 1.0f, 0.0f)
            : glm::vec3(0.0f, 0.0f, 1.0f);
        rightCandidate = glm::cross(camera.view, helper);
    }
    camera.right = normalizedOr(rightCandidate,
        glm::vec3(1.0f, 0.0f, 0.0f));
    camera.up = normalizedOr(glm::cross(camera.right, camera.view),
        referenceUp);
    cameraPosition = camera.position;
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
    cudaGLSetGLDevice(0);

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
    cudaGLRegisterBufferObject(pbo);
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

    PreviewOptions options;
    try
    {
        options = parsePreviewOptions(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n'
                  << previewUsage(argc > 0 ? argv[0] : "cis565_path_tracer");
        return 1;
    }
    if (options.help)
    {
        std::cout << previewUsage(argc > 0 ? argv[0] : "cis565_path_tracer");
        return 0;
    }

    RendererConfig rendererConfig;
    rendererConfig.sort = options.sort.value_or(rendererConfig.sort);
    rendererConfig.wavefront = options.wavefront.value_or(rendererConfig.wavefront);
    rendererConfig.rr = options.rr.value_or(rendererConfig.rr);
    rendererConfig.rrStart = options.rrStart.value_or(rendererConfig.rrStart);
    rendererConfig.sampler = options.sampler.value_or(rendererConfig.sampler);
    rendererConfig.motion = options.motion.value_or(rendererConfig.motion);
    if (rendererConfig.wavefront && rendererConfig.sort)
    {
        std::cerr << "error: --sort on requires --wavefront off\n";
        return 1;
    }

    const char* sceneFile = options.sceneFile.c_str();

    // Load the scene only after rejecting incompatible CLI scheduling flags.
    scene = new Scene(sceneFile);
    rendererConfig.maxDepth = scene->state.traceDepth > 0
        ? scene->state.traceDepth : rendererConfig.maxDepth;
    try
    {
        pathtraceSetConfig(rendererConfig);
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        delete scene;
        scene = nullptr;
        return 1;
    }

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;
    originalCamera = cam;
    cameraPosition = cam.position;

    const glm::vec3 offset = cam.position - cam.lookAt;
    zoom = glm::length(offset);
    if (!(zoom > 1.0e-6f) || !std::isfinite(zoom))
    {
        zoom = cam.focusDistance > 1.0e-6f && std::isfinite(cam.focusDistance)
            ? cam.focusDistance : 1.0f;
    }
    phi = std::atan2(offset.x, offset.z);
    theta = std::acos(std::clamp(offset.y / zoom, -1.0f, 1.0f));
    if (!std::isfinite(phi) || !std::isfinite(theta))
    {
        phi = 0.0f;
        theta = 0.5f * static_cast<float>(PI);
    }
    originalZoom = zoom;
    originalPhi = phi;
    originalTheta = theta;

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    if (pathTracerInitialized)
    {
        pathtraceFree();
        pathTracerInitialized = false;
    }

    return 0;
}

void saveImage()
{
    if (!pathTracerInitialized)
    {
        std::cerr << "Renderer is not initialized; no image was saved." << std::endl;
        return;
    }
    const std::uint64_t completedSamples = pathtraceSampleCount();
    if (completedSamples == 0)
    {
        std::cerr << "No completed samples to save." << std::endl;
        return;
    }

    std::vector<glm::vec3> accumulation;
    pathtraceDownloadAccumulation(accumulation);
    const std::size_t expectedPixels = static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);
    if (accumulation.size() != expectedPixels)
    {
        throw std::runtime_error("renderer returned an unexpected accumulation size");
    }

    const float samples = static_cast<float>(completedSamples);
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            const glm::vec3 pix = accumulation[static_cast<std::size_t>(index)];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename;
    std::error_code error;
    do
    {
        std::ostringstream name;
        name << renderState->imageName << "." << startTimeString << "."
             << completedSamples << "samp.save" << saveSequence++;
        filename = name.str();
        error.clear();
    }
    while (std::filesystem::exists(filename + ".png", error) && !error);
    if (error)
    {
        throw std::runtime_error("could not choose a unique preview filename: "
            + error.message());
    }
    img.savePNG(filename);
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        updateOrbitCamera(cam);
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree();
        pathTracerInitialized = false;
        pathtraceInit(scene);
        pathTracerInitialized = true;
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
        pathTracerInitialized = false;
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
                renderState = &scene->state;
                renderState->camera = originalCamera;
                zoom = originalZoom;
                phi = originalPhi;
                theta = originalTheta;
                cameraPosition = originalCamera.position;
                camchanged = false;
                iteration = 0;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    (void)mods;
    bool* pressed = nullptr;
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        pressed = &leftMousePressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        pressed = &rightMousePressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        pressed = &middleMousePressed;
    }
    if (pressed == nullptr)
    {
        return;
    }

    if (action == GLFW_RELEASE)
    {
        *pressed = false;
        if (!leftMousePressed && !rightMousePressed && !middleMousePressed)
        {
            cursorPositionValid = false;
        }
        return;
    }
    if (action == GLFW_PRESS && !MouseOverImGuiWindow())
    {
        *pressed = true;
        glfwGetCursorPos(window, &lastX, &lastY);
        cursorPositionValid = true;
    }
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    (void)window;
    if (!cursorPositionValid)
    {
        lastX = xpos;
        lastY = ypos;
        return;
    }
    if (xpos == lastX && ypos == lastY)
    {
        return;
    }

    const double deltaX = xpos - lastX;
    const double deltaY = ypos - lastY;
    lastX = xpos;
    lastY = ypos;

    if (leftMousePressed)
    {
        phi -= static_cast<float>(deltaX) / std::max(width, 1);
        theta -= static_cast<float>(deltaY) / std::max(height, 1);
        theta = std::clamp(theta, 1.0e-4f,
            static_cast<float>(PI) - 1.0e-4f);
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        const float scale = std::max(zoom, 1.0f);
        zoom += static_cast<float>(deltaY) * scale / std::max(height, 1);
        zoom = std::max(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        if (camchanged)
        {
            updateOrbitCamera(cam);
            camchanged = false;
        }
        const glm::vec3 view = normalizedOr(cam.view,
            glm::vec3(0.0f, 0.0f, -1.0f));
        const glm::vec3 right = normalizedOr(cam.right
            - view * glm::dot(cam.right, view),
            normalizedOr(glm::cross(view, cam.up),
                glm::vec3(1.0f, 0.0f, 0.0f)));
        const glm::vec3 up = normalizedOr(glm::cross(right, view),
            glm::vec3(0.0f, 1.0f, 0.0f));
        float worldPerPixelX = zoom * std::fabs(cam.pixelLength.x);
        float worldPerPixelY = zoom * std::fabs(cam.pixelLength.y);
        if (!(worldPerPixelX > 0.0f) || !std::isfinite(worldPerPixelX))
        {
            worldPerPixelX = zoom / std::max(width, 1);
        }
        if (!(worldPerPixelY > 0.0f) || !std::isfinite(worldPerPixelY))
        {
            worldPerPixelY = zoom / std::max(height, 1);
        }
        const glm::vec3 translation =
            -static_cast<float>(deltaX) * worldPerPixelX * right
            + static_cast<float>(deltaY) * worldPerPixelY * up;
        cam.position += translation;
        cam.lookAt += translation;
        cameraPosition = cam.position;
        iteration = 0;
    }
}
