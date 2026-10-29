#include "app/cli.h"
#include "app/glslUtility.hpp"
#include "app/image.h"
#include "render/pathtrace.h"
#include "scene/scene.h"
#include "scene/sceneStructs.h"
#include "timing.h"
#include "utilities.h"
#include "app/viewport_camera.h"

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string startTimeString;
static Options options;

static ViewportPose viewportPose;
static ViewportPose scenePose;  // the scene file's camera
static float flySpeed;          // world units per second
static bool camchanged = true;

// The default fly speed crosses the scene's bounding box diagonal in this time
static const float SECONDS_TO_CROSS_SCENE = 4.0f;
// RMB + wheel scales the fly speed by this per notch
static const float SPEED_STEP = 1.25f;
// A wheel notch moves as far as this much flying
static const float WHEEL_STEP_SECONDS = 0.25f;
// The longest fly step a stalled frame may take
static const float MAX_FRAME_SECONDS = 0.25f;

static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static Scene* scene;
static std::string sceneFile;
static std::string sceneTitle;
static GuiDataContainer guiData;
static RenderState* renderState;
static int iteration;

// Scene and environment loads run after the frame that asked, so the "Loading" line is on screen while they block.
static std::vector<SceneName> sceneList;
static int selectedScene = -1;
static char scenePath[1024] = "";
static std::string requestedScene;
static std::string loadError;

static std::vector<std::string> environmentList;
static int selectedEnvironment = -1;
static char environmentPath[1024] = "";
static bool environmentRequested = false;
static std::string requestedEnvironment;  // empty: the scene file's own
static std::string environmentError;

static std::vector<std::string> lensList;
static int selectedLens = -1;
static char lensPath[1024] = "";
static std::string lensError;

static int width;
static int height;

static GLuint positionLocation = 0;
static GLuint texcoordsLocation = 1;
static GLuint pbo;
static GLuint displayImage;

static GLFWwindow* window;
static ImGuiIO* io = nullptr;

static void runCuda();
static void runHeadless();
static void flyFromKeys(float seconds);
static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
static void framebufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight);
static void switchScene(const std::string& argument);
static void switchEnvironment(const std::string& file);
static void switchLens(const std::string& file);
static void selectLensInList();
static std::string defaultLensFile();
static void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

static std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

static void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

static void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    // v = 0 at the top of the window: the texture's first row is the image's top.
    GLfloat texcoords[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f
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

static GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

static void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

static void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

static void cleanupCuda()
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

static void initCuda()
{
    cudaGLSetGLDevice(0);
}

static void initPBO()
{
    int numTexels = width * height;
    int numValues = numTexels * 4;
    int sizeTexData = sizeof(GLubyte) * numValues;

    glGenBuffers(1, &pbo);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeTexData, NULL, GL_DYNAMIC_COPY);
    CUDA_CHECK(cudaGLRegisterBufferObject(pbo));
}

static void errorCallback(int error, const char* description)
{
    fprintf(stderr, "GLFW: %s\n", description);
}

// Creates the window, GL context, ImGui, the CUDA-mapped pixel buffer and the shader; false on failure.
static bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        return false;
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
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    if (glfwRawMouseMotionSupported())
    {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

// The panel's name for the environment that renders.
static std::string environmentLabel()
{
    if (!options.overrides.environmentFile.empty())
    {
        return std::filesystem::path(options.overrides.environmentFile).filename().string();
    }
    const EnvironmentSource& own = scene->environmentSource;
    if (!own.file.empty())
    {
        return std::filesystem::path(own.file).filename().string() + " (the scene's)";
    }
    return maxComponent(own.radiance) > 0.0f ? "one color (the scene's)" : "none";
}

static void renderImGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Path Tracer Analytics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Traced Depth %d", guiData.tracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Fly speed %.3g units/s (RMB + wheel)", flySpeed);

    ImGui::Separator();
    ImGui::Text("Scene: %s", sceneFile.c_str());
    ImGui::SetNextItemWidth(240);
    const char* preview = selectedScene >= 0 ? sceneList[selectedScene].name.c_str() : "pick a scene";
    if (ImGui::BeginCombo("##scene", preview))
    {
        for (int i = 0; i < (int)sceneList.size(); ++i)
        {
            const SceneName& entry = sceneList[i];
            const std::string label = entry.available ? entry.name : entry.name + " (not downloaded)";
            if (ImGui::Selectable(label.c_str(), i == selectedScene,
                    entry.available ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled))
            {
                selectedScene = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load") && selectedScene >= 0)
    {
        requestedScene = sceneList[selectedScene].name;
    }
    ImGui::SetNextItemWidth(240);
    if (ImGui::InputTextWithHint("##path", "or a .json / .gltf / .glb path, Enter loads", scenePath,
            sizeof scenePath, ImGuiInputTextFlags_EnterReturnsTrue) && scenePath[0] != '\0')
    {
        requestedScene = scenePath;
    }
    if (!requestedScene.empty())
    {
        ImGui::Text("Loading %s ...", requestedScene.c_str());
    }
    else if (!loadError.empty())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextColored(ImVec4(0.8f, 0.1f, 0.1f, 1.0f), "%s", loadError.c_str());
        ImGui::PopTextWrapPos();
    }

    ImGui::Separator();
    ImGui::Text("Environment: %s", environmentLabel().c_str());
    ImGui::SetNextItemWidth(240);
    const std::string environmentPreview = selectedEnvironment >= 0
        ? std::filesystem::path(environmentList[selectedEnvironment]).filename().string()
        : environmentList.empty() ? "no maps in " + ENVIRONMENT_DIR : "pick a map";
    if (ImGui::BeginCombo("##environment", environmentPreview.c_str()))
    {
        for (int i = 0; i < (int)environmentList.size(); ++i)
        {
            const std::string name = std::filesystem::path(environmentList[i]).filename().string();
            if (ImGui::Selectable(name.c_str(), i == selectedEnvironment))
            {
                selectedEnvironment = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load##environment") && selectedEnvironment >= 0)
    {
        environmentRequested = true;
        requestedEnvironment = environmentList[selectedEnvironment];
    }
    ImGui::SetNextItemWidth(240);
    if (ImGui::InputTextWithHint("##environmentPath", "or a .hdr / .exr path, Enter loads", environmentPath,
            sizeof environmentPath, ImGuiInputTextFlags_EnterReturnsTrue) && environmentPath[0] != '\0')
    {
        environmentRequested = true;
        requestedEnvironment = environmentPath;
    }
    if (!options.overrides.environmentFile.empty() && ImGui::Button("Back to the scene's own"))
    {
        environmentRequested = true;
        requestedEnvironment.clear();
    }
    if (environmentRequested)
    {
        ImGui::Text("Loading %s ...", requestedEnvironment.empty() ? "the scene's environment"
                                                                   : requestedEnvironment.c_str());
    }
    else if (!environmentError.empty())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextColored(ImVec4(0.8f, 0.1f, 0.1f, 1.0f), "%s", environmentError.c_str());
        ImGui::PopTextWrapPos();
    }

    ImGui::Separator();
    const bool neeAvailable = pathtraceNeeAvailable();
    ImGui::BeginDisabled(!neeAvailable);
    if (ImGui::Checkbox("Next event estimation (NEE + MIS)", &options.nee))
    {
        setNextEventEstimation(options.nee);
        camchanged = true;
    }
    ImGui::EndDisabled();
    if (!neeAvailable)
    {
        ImGui::TextDisabled("needs OptiX and a light to sample");
    }

    ImGui::Separator();
    Camera& cam = renderState->camera;
    const char* cameraNames[] = {"Pinhole", "Thin lens", "Real lens"};
    ImGui::SetNextItemWidth(240);
    if (ImGui::BeginCombo("Camera", cameraNames[cam.type]))
    {
        for (int i = 0; i < 3; ++i)
        {
            const bool allowed = i != CAMERA_REAL_LENS || scene->hasLens() || !lensList.empty();
            if (ImGui::Selectable(cameraNames[i], i == (int)cam.type, allowed ? 0 : ImGuiSelectableFlags_Disabled)
                && i != (int)cam.type)
            {
                if (i == CAMERA_REAL_LENS && !scene->hasLens())
                {
                    switchLens(defaultLensFile());  // selects the real lens itself when the file loads
                }
                else
                {
                    cam.type = (CameraType)i;
                    camchanged = true;
                }
            }
        }
        ImGui::EndCombo();
    }
    if (cam.type != CAMERA_PINHOLE)
    {
        // Drag steps scale with the scene: a radius that blurs a room is a wall for a figurine.
        const float step = 0.001f * viewportPose.pivotDistance;
        if (cam.type == CAMERA_THIN_LENS)
        {
            ImGui::SetNextItemWidth(240);
            if (ImGui::DragFloat("Aperture radius", &cam.apertureRadius, step, 0.0f, FLT_MAX, "%.4g"))
            {
                camchanged = true;
            }
        }
        ImGui::SetNextItemWidth(240);
        if (ImGui::DragFloat("Focus distance", &cam.focusDistance, 10.0f * step, 10.0f * step, FLT_MAX, "%.4g"))
        {
            if (cam.type == CAMERA_REAL_LENS)
            {
                pathtraceSetLens();
            }
            camchanged = true;
        }
    }
    ImGui::Text("Lens: %s", scene->hasLens() ? scene->lens.name.c_str() : "none");
    ImGui::SetNextItemWidth(240);
    const std::string lensPreview = selectedLens >= 0
        ? std::filesystem::path(lensList[selectedLens]).filename().string()
        : lensList.empty() ? "no lens files in " + LENS_DIR : "pick a lens";
    if (ImGui::BeginCombo("##lens", lensPreview.c_str()))
    {
        for (int i = 0; i < (int)lensList.size(); ++i)
        {
            const std::string name = std::filesystem::path(lensList[i]).filename().string();
            if (ImGui::Selectable(name.c_str(), i == selectedLens))
            {
                selectedLens = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load##lens") && selectedLens >= 0)
    {
        switchLens(lensList[selectedLens]);
    }
    ImGui::SetNextItemWidth(240);
    if (ImGui::InputTextWithHint("##lensPath", "or a .json path, Enter loads", lensPath, sizeof lensPath,
            ImGuiInputTextFlags_EnterReturnsTrue) && lensPath[0] != '\0')
    {
        switchLens(lensPath);
    }
    if (!lensError.empty())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextColored(ImVec4(0.8f, 0.1f, 0.1f, 1.0f), "%s", lensError.c_str());
        ImGui::PopTextWrapPos();
    }
    if (scene->hasLens())
    {
        const LensSystem::Paraxial& p = scene->lens.paraxial;
        ImGui::TextDisabled("%d surfaces, f = %.1f mm, f/%.2f", (int)scene->lens.surfaces.size(), p.focalLength,
            p.fNumber);
        if (cam.type == CAMERA_REAL_LENS)
        {
            if (ImGui::Checkbox("Exit pupil sampling", &options.exitPupil))
            {
                setExitPupil(options.exitPupil);
                camchanged = true;
            }
            if (ImGui::Checkbox("Physical camera weight", &options.physicalCameraWeight))
            {
                setPhysicalCameraWeight(options.physicalCameraWeight);
                camchanged = true;
            }
            ImGui::TextDisabled("film %.3f mm behind the front vertex", pathtraceLensFilmDistance());
            ImGui::TextDisabled("exit pupil: %d bins built in %.2f ms", pathtraceExitPupilBins(), pathtraceExitPupilBuildMs());
            ImGui::TextDisabled("blocked %.1f%% of camera rays", 100.0f * pathtraceLensBlockedFraction());
        }
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

static void mainLoop()
{
    double lastFrameTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        const bool finished = !camchanged && iteration >= (int)renderState->iterations;
        if (finished)
        {
            glfwWaitEventsTimeout(0.1);
        }
        else
        {
            glfwPollEvents();
        }

        const double now = glfwGetTime();
        flyFromKeys((float)std::min(now - lastFrameTime, (double)MAX_FRAME_SECONDS));
        lastFrameTime = now;

        runCuda();

        std::string title = "CIS565 Path Tracer | " + sceneTitle + " | " + std::to_string(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        renderImGui();

        glfwSwapBuffers(window);

        if (!requestedScene.empty())
        {
            switchScene(requestedScene);
            requestedScene.clear();
        }
        if (environmentRequested)
        {
            switchEnvironment(requestedEnvironment);
            environmentRequested = false;
        }
    }

    pathtraceFree();
    cleanupCuda();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    cudaDeviceReset();
}

// Makes s the scene being rendered; freeing the old scene is the caller's job.
static void useScene(Scene* s, const std::string& file)
{
    scene = s;
    sceneFile = file;
    sceneTitle = std::filesystem::path(file).stem().string();
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    // Headless renders take their basis from this pose too, so they match the window's first frame.
    scenePose = poseFromCamera(cam);
    viewportPose = scenePose;
    applyPose(viewportPose, cam);
    camchanged = true;

    glm::vec3 lo, hi;
    scene->bounds(0, scene->geoms.size(), lo, hi);
    flySpeed = (lo.x <= hi.x ? glm::length(hi - lo) : 1.0f) / SECONDS_TO_CROSS_SCENE;
}

// Replaces the window's scene; one that fails to load leaves the old scene in place.
static void switchScene(const std::string& argument)
{
    Scene* next;
    std::string file;
    try
    {
        file = findSceneFile(argument);
        next = new Scene(file, options.overrides);
    }
    catch (const std::exception& e)
    {
        loadError = e.what();
        fprintf(stderr, "error: %s\n", loadError.c_str());
        return;
    }
    loadError.clear();
    selectedScene = -1;
    for (int i = 0; i < (int)sceneList.size(); ++i)
    {
        if (sceneList[i].name == argument)
        {
            selectedScene = i;
        }
    }

    pathtraceFree();
    delete scene;
    useScene(next, file);
    pathtraceInit(scene);

    cleanupCuda();
    initTextures();
    initPBO();
    glfwSetWindowSize(window, width, height);
}

static void selectEnvironmentInList()
{
    selectedEnvironment = -1;
    for (int i = 0; i < (int)environmentList.size(); ++i)
    {
        if (environmentList[i] == options.overrides.environmentFile)
        {
            selectedEnvironment = i;
        }
    }
}

static std::string defaultLensFile()
{
    for (const std::string& file : lensList)
    {
        if (std::filesystem::path(file).filename() == "dgauss.50mm.json")
        {
            return file;
        }
    }
    return lensList.empty() ? std::string() : lensList.front();
}

static void selectLensInList()
{
    selectedLens = -1;
    for (int i = 0; i < (int)lensList.size(); ++i)
    {
        if (scene->hasLens() && lensList[i] == std::filesystem::path(scene->lens.file).generic_string())
        {
            selectedLens = i;
        }
    }
}

// Loads a lens file and renders through it; a bad file keeps the current lens.
static void switchLens(const std::string& file)
{
    try
    {
        scene->lens = loadLens(file);
    }
    catch (const std::exception& e)
    {
        lensError = e.what();
        fprintf(stderr, "error: %s\n", lensError.c_str());
        return;
    }
    lensError.clear();
    options.overrides.lens = file;  // later scene switches keep it
    selectLensInList();
    pathtraceSetLens();
    renderState->camera.type = CAMERA_REAL_LENS;
    camchanged = true;
}

// Replaces the environment with a map file, or the scene's own when file is empty; a failed load keeps the current one.
static void switchEnvironment(const std::string& file)
{
    try
    {
        scene->environment = loadEnvironment(file.empty()
                ? scene->environmentSource
                : EnvironmentSource{ file, glm::vec3(1.0f), 0.0f },
            options.overrides.environmentCompensation);
    }
    catch (const std::exception& e)
    {
        environmentError = e.what();
        fprintf(stderr, "error: %s\n", environmentError.c_str());
        return;
    }
    environmentError.clear();
    options.overrides.environmentFile = file;  // later scene switches keep it
    selectEnvironmentInList();
    selectLensInList();
    pathtraceSetEnvironment(scene->environment);
    camchanged = true;
}

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    options = parseArguments(argc, argv);
    setTiming(options.timing);
    setRussianRoulette(options.russianRoulette);
    setAntialiasing(options.antialiasing);
    setMaterialSort(options.materialSort);
    setCompaction(options.compaction);
    setExitPupil(options.exitPupil);
    setPhysicalCameraWeight(options.physicalCameraWeight);
    setNextEventEstimation(options.nee);
    setOptix(options.optix);
    setOptixValidation(options.optixValidation);
    setToneMap(options.toneMap, options.exposure);

    try
    {
        if (options.list)
        {
            listScenes();
            return 0;
        }
        const std::string file = findSceneFile(options.sceneFile);
        useScene(new Scene(file, options.overrides), file);
        if (!options.headless)
        {
            sceneList = sceneNames();
            environmentList = environmentMapFiles();
            lensList = lensFiles();
        }
    }
    catch (const std::exception& e)
    {
        fatal("%s", e.what());
    }

    if (options.headless)
    {
        runHeadless();
        return 0;
    }

    for (int i = 0; i < (int)sceneList.size(); ++i)
    {
        if (sceneList[i].name == options.sceneFile)
        {
            selectedScene = i;
        }
    }
    selectEnvironmentInList();
    selectLensInList();

    if (!init())
    {
        fatal("cannot create the window and its OpenGL context");
    }
    // Device buffers live until exit or a scene switch; a camera change only clears the image.
    pathtraceInit(scene);
    setGuiData(&guiData);

    mainLoop();

    return 0;
}

static void saveImage()
{
    float samples = iteration;
    pathtraceDownloadImage();
    Image img(width, height);
    const std::string ext = lowercaseExtension(options.outPath);
    const bool hdr = ext == ".hdr";
    const bool exr = ext == ".exr";

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index] / samples;
            img.setPixel(x, y, hdr || exr ? pix : applyToneMap(pix, options.toneMap, options.exposure));
        }
    }

    std::string filename;
    if (!options.outPath.empty())
    {
        // savePNG, saveHDR and saveEXR append the extension themselves
        filename = options.outPath;
        if (filename.size() > 4 && (filename.substr(filename.size() - 4) == ".png" || hdr || exr))
        {
            filename = filename.substr(0, filename.size() - 4);
        }
    }
    else
    {
        filename = "img/auto_saved/" + renderState->imageName;
        std::ostringstream ss;
        ss << filename << "." << startTimeString << "." << samples << "samp";
        filename = ss.str();
    }

    std::filesystem::path dir = std::filesystem::path(filename).parent_path();
    if (!dir.empty())
    {
        std::filesystem::create_directories(dir);
    }

    if (hdr)
    {
        img.saveHDR(filename);
    }
    else if (exr)
    {
        img.saveEXR(filename);
    }
    else
    {
        img.savePNG(filename);
    }
}

// Renders the scene's iterations without a window, saves and returns.
static void runHeadless()
{
    pathtraceInit(scene);

    auto t0 = std::chrono::steady_clock::now();
    for (iteration = 1; iteration <= renderState->iterations; iteration++)
    {
        pathtrace(nullptr, iteration);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    iteration = renderState->iterations;
    {
        TimingScope timing("save.download_tonemap_png");
        saveImage();
    }
    if (options.timing)
    {
        timingAdd("render.spp", renderState->iterations);
        timingAdd("render.total", ms);
        timingReport();
        pathtraceTimingReport();
    }
    pathtraceFree();
    cudaDeviceReset();

    printf("headless: %dx%d, %d spp, %.1f ms total, %.2f ms/spp\n",
        width, height, renderState->iterations, ms, ms / renderState->iterations);
}

static void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        camchanged = false;
    }

    if (iteration == 0)
    {
        pathtraceReset();
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pboPointer = NULL;
        iteration++;
        CUDA_CHECK(cudaGLMapBufferObject((void**)&pboPointer, pbo));

        pathtrace(pboPointer, iteration);

        CUDA_CHECK(cudaGLUnmapBufferObject(pbo));

        if (iteration == (int)renderState->iterations)
        {
            saveImage();
        }
    }
}

// Every viewportPose change goes through here: the render camera follows and the image starts over.
static void cameraMoved()
{
    applyPose(viewportPose, renderState->camera);
    camchanged = true;
}

static void framebufferSizeCallback(GLFWwindow* window, int fbWidth, int fbHeight)
{
    glViewport(0, 0, fbWidth, fbHeight);
}

static bool altHeld()
{
    return glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
}

// Keys are polled every frame so a held key moves smoothly; not with Alt, where RMB dollies.
static void flyFromKeys(float seconds)
{
    if (!rightMousePressed || altHeld() || io->WantCaptureKeyboard)
    {
        return;
    }
    auto held = [](int key) { return glfwGetKey(window, key) == GLFW_PRESS ? 1 : 0; };
    if (fly(viewportPose, renderState->camera, held(GLFW_KEY_W) - held(GLFW_KEY_S), held(GLFW_KEY_D) - held(GLFW_KEY_A),
            held(GLFW_KEY_E) - held(GLFW_KEY_Q), flySpeed * seconds))
    {
        cameraMoved();
    }
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS || io->WantCaptureKeyboard)
    {
        return;
    }
    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            saveImage();
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            // Plain S flies backward
            if (mods & GLFW_MOD_CONTROL)
            {
                saveImage();
            }
            break;
        case GLFW_KEY_F:
            // Not while flying, where F sits next to the movement keys
            if (!rightMousePressed)
            {
                viewportPose = scenePose;
                cameraMoved();
            }
            break;
    }
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    const bool press = action == GLFW_PRESS;
    // Releases always count, so a drag that ends over a panel does not leave its button held.
    if (press && io->WantCaptureMouse)
    {
        return;
    }
    switch (button)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            leftMousePressed = press;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            rightMousePressed = press;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            middleMousePressed = press;
            break;
        default:
            return;
    }
    const bool dragging = leftMousePressed || rightMousePressed || middleMousePressed;
    glfwSetInputMode(window, GLFW_CURSOR, dragging ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    glfwGetCursorPos(window, &lastX, &lastY);
}

static void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    const double dx = xpos - lastX;
    const double dy = ypos - lastY;
    lastX = xpos;
    lastY = ypos;
    if (dx == 0.0 && dy == 0.0)
    {
        return;
    }

    const bool mirrored = renderState->camera.mirrored;
    const bool alt = altHeld();
    if (alt && leftMousePressed)
    {
        orbit(viewportPose, dx, dy, mirrored);
    }
    else if (alt && rightMousePressed)
    {
        dollyToPivot(viewportPose, dx, dy);
    }
    else if (middleMousePressed || (leftMousePressed && rightMousePressed))
    {
        pan(viewportPose, renderState->camera, dx, dy);
    }
    else if (rightMousePressed)
    {
        turn(viewportPose, dx, dy, mirrored);
    }
    else if (leftMousePressed)
    {
        moveAlongGround(viewportPose, dx, dy, flySpeed, mirrored);
    }
    else
    {
        return;
    }
    cameraMoved();
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (io->WantCaptureMouse)
    {
        return;
    }
    if (rightMousePressed)
    {
        flySpeed *= std::pow(SPEED_STEP, (float)yoffset);
        return;
    }
    moveAlongView(viewportPose, (float)yoffset * flySpeed * WHEEL_STEP_SECONDS);
    cameraMoved();
}
