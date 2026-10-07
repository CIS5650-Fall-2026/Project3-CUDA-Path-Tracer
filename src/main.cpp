#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "render_checkpoint.h"
#include "checkpoint_build_id.h"
#include "json.hpp"
#include <thrust/version.h>

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
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <iomanip>

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
static bool rendererInitialized = false;
static bool restoreAccumulation = false;
static std::string checkpointPath;
static int checkpointAt = 0; // Optional exact boundary: save and exit.
static std::string statsPrefix;
using StatsClock = std::chrono::steady_clock;
struct IterationStat {
    int sample;
    double milliseconds;
    bool sorted, compacted, antialiased;
};
struct CheckpointStat {
    std::string operation, file;
    int sample;
    double milliseconds;
    std::uintmax_t bytes;
};
static std::vector<IterationStat> iterationStats;
static std::vector<PathTraceBounce> diagnosticBounces;
static std::vector<CheckpointStat> checkpointStats;
static constexpr int diagnosticSample = 32;

static double elapsedMilliseconds(StatsClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(StatsClock::now() - start).count();
}

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

static std::string renderingBuildIdentity()
{
    return std::string(CHECKPOINT_SOURCE_ID) + ":" + CHECKPOINT_SETTINGS_ID + ":" +
        CHECKPOINT_BUILD_CONFIG + ":glm=" + std::to_string(GLM_VERSION) +
        ":thrust=" + std::to_string(THRUST_VERSION) + ":cuda=" + std::to_string(CUDART_VERSION);
}

static std::array<float, 3> fields(const glm::vec3& v) { return {v.x, v.y, v.z}; }
static glm::vec3 vector3(const std::array<float, 3>& v) { return glm::vec3(v[0], v[1], v[2]); }

static checkpoint::Data captureCheckpoint()
{
    checkpoint::Data d;
    const Camera& c = renderState->camera;
    d.width = c.resolution.x; d.height = c.resolution.y;
    d.traceDepth = renderState->traceDepth;
    d.completedSamples = iteration;
    d.sortMaterials = renderState->sortMaterials;
    d.sceneIdentity = scene->checkpointSceneIdentity;
    d.buildIdentity = renderingBuildIdentity();
    d.camera.position = fields(c.position); d.camera.lookAt = fields(c.lookAt);
    d.camera.view = fields(c.view); d.camera.up = fields(c.up); d.camera.right = fields(c.right);
    d.camera.fov = {c.fov.x, c.fov.y};
    d.camera.pixelLength = {c.pixelLength.x, c.pixelLength.y};
    d.controller.zoom = zoom; d.controller.theta = theta; d.controller.phi = phi;
    d.controller.originalLookAt = fields(ogLookAt);
    d.controller.cameraPosition = fields(cameraPosition);
    d.accumulation.reserve(renderState->image.size() * 3);
    for (const glm::vec3& pixel : renderState->image) {
        d.accumulation.push_back(pixel.x);
        d.accumulation.push_back(pixel.y);
        d.accumulation.push_back(pixel.z);
    }
    return d;
}

static void restoreCheckpoint(const checkpoint::Data& d)
{
    Camera& c = renderState->camera;
    checkpoint::requireCompatible(d, {
        static_cast<std::uint32_t>(c.resolution.x), static_cast<std::uint32_t>(c.resolution.y),
        static_cast<std::uint32_t>(renderState->traceDepth),
        scene->checkpointSceneIdentity, renderingBuildIdentity()});
    if (renderState->iterations < d.completedSamples) {
        throw std::runtime_error("Target samples are below checkpoint count; increase --samples");
    }
    // Restore exact float bits; rebuilding the basis would perturb future rays.
    c.resolution = glm::ivec2(d.width, d.height);
    c.position = vector3(d.camera.position); c.lookAt = vector3(d.camera.lookAt);
    c.view = vector3(d.camera.view); c.up = vector3(d.camera.up); c.right = vector3(d.camera.right);
    c.fov = glm::vec2(d.camera.fov[0], d.camera.fov[1]);
    c.pixelLength = glm::vec2(d.camera.pixelLength[0], d.camera.pixelLength[1]);
    zoom = d.controller.zoom; theta = d.controller.theta; phi = d.controller.phi;
    ogLookAt = vector3(d.controller.originalLookAt);
    cameraPosition = vector3(d.controller.cameraPosition);
    renderState->traceDepth = static_cast<int>(d.traceDepth);
    renderState->sortMaterials = d.sortMaterials;
    for (std::size_t i = 0; i < renderState->image.size(); ++i) {
        renderState->image[i] = glm::vec3(
            d.accumulation[3 * i], d.accumulation[3 * i + 1], d.accumulation[3 * i + 2]);
    }
    iteration = static_cast<int>(d.completedSamples);
    restoreAccumulation = true;
    camchanged = false;
    leftMousePressed = rightMousePressed = middleMousePressed = false;
}

static void saveCheckpointAtBoundary()
{
    if (camchanged || iteration <= 0) {
        throw std::runtime_error("Checkpoint needs at least one completed sample of the current camera");
    }
    // pathtrace() synchronously copies dev_image to state.image before returning
    // Called only outside pathtrace() after the count has been committed
    const auto started = statsPrefix.empty() ? StatsClock::time_point{} : StatsClock::now();
    checkpoint::write(checkpointPath, captureCheckpoint());
    if (!statsPrefix.empty()) {
        const double ms = elapsedMilliseconds(started);
        checkpointStats.push_back({"save_capture_write", checkpointPath, iteration,
            ms, std::filesystem::file_size(checkpointPath)});
    }
    std::cout << "Checkpoint: " << checkpointPath << " (" << iteration << " completed samples)\n";
}

static int positiveCount(const std::string& value)
{
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Sample counts must be positive integers");
    const auto n = std::stoull(value);
    if (n == 0 || n > std::numeric_limits<int>::max())
        throw std::runtime_error("Sample count must be in [1, INT_MAX]");
    return static_cast<int>(n);
}

static void cudaOrThrow(cudaError_t e, const char* operation)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}

// Buffered output: no file/console I/O occurs inside the timed pathtrace call.
static void writeStatistics()
{
    if (statsPrefix.empty()) return;
    auto output = [](const std::string& path) {
        std::ofstream file(path);
        if (!file) throw std::runtime_error("Cannot write statistics: " + path);
        file << std::setprecision(17);
        return file;
    };
    auto iterations = output(statsPrefix + ".iterations.csv");
    iterations << "sample,pathtrace_ms,sort_materials,compact_paths,antialiasing\n";
    for (const auto& r : iterationStats)
        iterations << r.sample << ',' << r.milliseconds << ',' << r.sorted << ','
                   << r.compacted << ',' << r.antialiased << '\n';
    auto bounces = output(statsPrefix + ".active_paths.csv");
    bounces << "sample,bounce,active_after,launched_paths\n";
    for (const auto& r : diagnosticBounces) {
        bounces << diagnosticSample << ',' << r.bounce << ',' << r.activeAfter
                << ',' << r.launchedPaths << '\n';
        std::cout << "Sample " << diagnosticSample << ", bounce " << r.bounce
                  << ": active=" << r.activeAfter << ", launched=" << r.launchedPaths << '\n';
    }
    nlohmann::json metadata;
    metadata["schema_version"] = 1;
    metadata["build_identity"] = renderingBuildIdentity();
    metadata["build_configuration"] = CHECKPOINT_BUILD_CONFIG;
    metadata["scene_identity"] = scene->checkpointSceneIdentity;
    metadata["width"] = width; metadata["height"] = height;
    metadata["depth"] = renderState->traceDepth;
    metadata["completed_samples"] = iteration;
    metadata["target_samples"] = renderState->iterations;
    metadata["sort_materials"] = renderState->sortMaterials;
    metadata["compact_paths"] = renderState->compactPaths;
    metadata["antialiasing"] = renderState->antialiasing;
    metadata["diagnostic_sample"] = diagnosticSample;
    metadata["cuda_runtime"] = CUDART_VERSION;
    metadata["glm_version"] = GLM_VERSION;
    metadata["thrust_version"] = THRUST_VERSION;
    metadata["timing_scope"] = "pathtrace wall time including PBO write, RGB download and ERRORCHECK synchronization; excludes map/unmap, UI, initialization, file output";
    int device = 0, driver = 0;
    cudaDeviceProp properties{};
    cudaOrThrow(cudaGetDevice(&device), "statistics device");
    cudaOrThrow(cudaGetDeviceProperties(&properties, device), "statistics device properties");
    cudaOrThrow(cudaDriverGetVersion(&driver), "statistics driver version");
    metadata["gpu"] = properties.name;
    metadata["cuda_driver_api_version"] = driver;
    metadata["compute_capability"] = {properties.major, properties.minor};
    metadata["checkpoint_io"] = nlohmann::json::array();
    for (const auto& r : checkpointStats)
        metadata["checkpoint_io"].push_back({{"operation", r.operation}, {"file", r.file},
            {"completed_samples", r.sample}, {"milliseconds", r.milliseconds}, {"bytes", r.bytes}});
    auto meta = output(statsPrefix + ".metadata.json");
    meta << metadata.dump(2) << '\n';
    iterations.flush(); bounces.flush(); meta.flush();
    if (!iterations || !bounces || !meta) throw std::runtime_error("Statistics write failed");
}

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
    ImGui::Checkbox("Sort by material", &renderState->sortMaterials);
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
        if (glfwWindowShouldClose(window)) break;

        runCuda();

        std::string title = "Jacob Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
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

    pathtraceFree();
    rendererInitialized = false;
    cleanupCuda(); // Release interop while the GL context still exists.
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

    if (argc < 2) {
        printf("Usage: %s SCENE.json [--resume FILE.ptc] [--samples TOTAL]\n"
               "       [--checkpoint FILE.ptc] [--checkpoint-at N] [--stats PREFIX]\n", argv[0]);
        return 1;
    }

    try {
    std::string resumePath;
    int targetSamples = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing option value: " + option);
        const std::string value = argv[++i];
        if (option == "--resume") resumePath = value;
        else if (option == "--checkpoint") checkpointPath = value;
        else if (option == "--samples") targetSamples = positiveCount(value);
        else if (option == "--checkpoint-at") checkpointAt = positiveCount(value);
        else if (option == "--stats") statsPrefix = value;
        else throw std::runtime_error("Unknown option: " + option);
    }

    if (!statsPrefix.empty()) {
        const auto parent = std::filesystem::path(statsPrefix).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        iterationStats.reserve(1024);
    }

    const char* sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    if (targetSamples) renderState->iterations = static_cast<unsigned int>(targetSamples);
    if (checkpointPath.empty()) checkpointPath = resumePath.empty()
        ? renderState->imageName + ".ptc" : resumePath;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    if (!resumePath.empty()) {
        const auto started = statsPrefix.empty() ? StatsClock::time_point{} : StatsClock::now();
        restoreCheckpoint(checkpoint::read(resumePath));
        if (!statsPrefix.empty()) {
            const double ms = elapsedMilliseconds(started);
            checkpointStats.push_back({"read_host_restore", resumePath, iteration,
                ms, std::filesystem::file_size(resumePath)});
        }
        std::cout << "Resuming after sample " << iteration << "; next sample is " << iteration + 1ll << '\n';
    }
    if (checkpointAt && (checkpointAt < iteration ||
        static_cast<unsigned int>(checkpointAt) > renderState->iterations)) {
        throw std::runtime_error("--checkpoint-at must be between the completed count and target");
    }

    // Initialize CUDA and GL components
    if (!init()) throw std::runtime_error("Could not initialize the preview");
    glfwGetCursorPos(window, &lastX, &lastY);

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();
    writeStatistics();

    delete guiData;
    delete scene;
    return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Path tracer: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

void saveImage()
{
    if (iteration <= 0) {
        std::cout << "No completed samples to export.\n";
        return;
    }
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << iteration << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        restoreAccumulation = false;
        if (rendererInitialized) {
            pathtraceFree();
            rendererInitialized = false;
        }
        std::fill(renderState->image.begin(), renderState->image.end(), glm::vec3(0.0f));
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (!rendererInitialized)
    {
        const bool restoring = restoreAccumulation;
        const auto started = statsPrefix.empty() ? StatsClock::time_point{} : StatsClock::now();
        pathtraceInit(scene, restoreAccumulation);
        if (!statsPrefix.empty())
            checkpointStats.push_back({restoring ? "restore_gpu_init" : "fresh_gpu_init",
                "", iteration, elapsedMilliseconds(started), 0});
        rendererInitialized = true;
        restoreAccumulation = false;
        // Also display the restored image when its target is already reached.
        uchar4* restoredPbo = nullptr;
        cudaOrThrow(cudaGLMapBufferObject((void**)&restoredPbo, pbo), "map initial preview");
        pathtraceDisplay(restoredPbo, iteration);
        cudaOrThrow(cudaGLUnmapBufferObject(pbo), "unmap initial preview");
    }

    if (checkpointAt != 0 && iteration == checkpointAt) {
        saveCheckpointAtBoundary();
        glfwSetWindowShouldClose(window, GL_TRUE);
        return;
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        cudaOrThrow(cudaGLMapBufferObject((void**)&pbo_dptr, pbo), "map preview");

        // execute the kernel
        int frame = 0;
        std::vector<PathTraceBounce>* counts = !statsPrefix.empty() && iteration + 1 == diagnosticSample
            ? &diagnosticBounces : nullptr;
        const auto started = statsPrefix.empty() ? StatsClock::time_point{} : StatsClock::now();
        pathtrace(pbo_dptr, frame, iteration + 1, counts);
        if (!statsPrefix.empty())
            iterationStats.push_back({iteration + 1, elapsedMilliseconds(started),
                renderState->sortMaterials, renderState->compactPaths, renderState->antialiasing});

        // unmap buffer object
        cudaOrThrow(cudaGLUnmapBufferObject(pbo), "unmap preview");
        ++iteration; // Commit only after a full iteration and host image copy.
        if (checkpointAt != 0 && iteration == checkpointAt) {
            saveCheckpointAtBoundary();
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }
    else
    {
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
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
            case GLFW_KEY_C:
                // GLFW dispatches this callback during glfwPollEvents(),
                // between complete pathtrace calls. Save before any close event.
                try { saveCheckpointAtBoundary(); }
                catch (const std::exception& error) {
                    std::cerr << "Checkpoint not saved: " << error.what() << '\n';
                }
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
