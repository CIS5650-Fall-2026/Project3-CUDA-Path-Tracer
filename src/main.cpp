#include "denoise.h"
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
#include <cstring>
#include <iostream>
#include <fstream>
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
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;
static int resumeIteration = 0;
static int firstIteration = 0;

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
    bool changed = ImGui::Checkbox("Sort by material", &imguiData->sortByMaterial);
    changed |= ImGui::Checkbox("Stream compaction", &imguiData->streamCompaction);
    changed |= ImGui::Checkbox("Antialiasing", &imguiData->antialiasing);
    changed |= ImGui::Checkbox("Russian roulette", &imguiData->russianRoulette);
    changed |= ImGui::Checkbox("Direct lighting", &imguiData->directLighting);
    changed |= ImGui::Checkbox("Halton sampling", &imguiData->halton);
    changed |= ImGui::Checkbox("BVH", &imguiData->useBvh);
    changed |= ImGui::Checkbox("Bounding box culling", &imguiData->cullBounds);
    changed |= ImGui::Checkbox("Denoise", &imguiData->denoise);
    changed |= ImGui::SliderFloat("Lens radius", &renderState->camera.lensRadius, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Focal distance", &renderState->camera.focalDistance, 0.1f, 30.0f);
    if (changed)
    {
        camchanged = true;
    }
    float samples = iteration > firstIteration ? iteration - firstIteration : 1;
    ImGui::Text("Per iteration (ms): total %.2f, intersect %.2f, sort %.2f, shade %.2f, compact %.2f, denoise %.2f",
        imguiData->msTotal / samples, imguiData->msIntersect / samples, imguiData->msSort / samples,
        imguiData->msShade / samples, imguiData->msCompact / samples, imguiData->msDenoise / samples);
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

struct CheckpointHeader
{
    int width;
    int height;
    int iteration;
    float zoom;
    float theta;
    float phi;
    glm::vec3 lookAt;
};

void saveCheckpoint(const std::string& filename)
{
    pathtraceDownload();
    CheckpointHeader header = { width, height, iteration, zoom, theta, phi, renderState->camera.lookAt };
    std::ofstream file(filename, std::ios::binary);
    file.write((const char*)&header, sizeof(header));
    for (const std::vector<glm::vec3>* buffer : { &renderState->image, &renderState->albedo, &renderState->normal })
    {
        file.write((const char*)buffer->data(), buffer->size() * sizeof(glm::vec3));
    }
    std::cout << "Saved " << filename << "." << std::endl;
}

void loadCheckpoint(const std::string& filename)
{
    CheckpointHeader header;
    std::ifstream file(filename, std::ios::binary);
    file.read((char*)&header, sizeof(header));
    if (!file || header.width != width || header.height != height)
    {
        printf("Couldn't resume from %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    for (std::vector<glm::vec3>* buffer : { &renderState->image, &renderState->albedo, &renderState->normal })
    {
        file.read((char*)buffer->data(), buffer->size() * sizeof(glm::vec3));
    }
    resumeIteration = header.iteration;
    zoom = header.zoom;
    theta = header.theta;
    phi = header.phi;
    renderState->camera.lookAt = header.lookAt;
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json [CHECKPOINT.ckpt] [options]\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();
    std::string checkpointFile;
    for (int i = 2; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg[0] != '-')
        {
            checkpointFile = arg;
        }
        else if (arg == "-denoise")
        {
            guiData->denoise = true;
        }
        else if (arg == "-sort")
        {
            guiData->sortByMaterial = true;
        }
        else if (arg == "-nocompact")
        {
            guiData->streamCompaction = false;
        }
        else if (arg == "-noaa")
        {
            guiData->antialiasing = false;
        }
        else if (arg == "-rr")
        {
            guiData->russianRoulette = true;
        }
        else if (arg == "-direct")
        {
            guiData->directLighting = true;
        }
        else if (arg == "-halton")
        {
            guiData->halton = true;
        }
        else if (arg == "-nobvh")
        {
            guiData->useBvh = false;
        }
        else if (arg == "-nocull")
        {
            guiData->cullBounds = false;
        }
    }

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
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
    glm::vec3 offset = cam.position - cam.lookAt;
    phi = atan2(offset.x, offset.z);
    theta = glm::acos(offset.y / glm::length(offset));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);
    if (!checkpointFile.empty())
    {
        loadCheckpoint(checkpointFile);
    }

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
    float samples = iteration;
    if (guiData->denoise)
    {
        pathtraceDownload();
        denoise(*renderState, iteration);
    }

    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = guiData->denoise ? renderState->denoised[index] : renderState->image[index] / samples;
            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
    saveCheckpoint(filename + ".ckpt");

    float timed = iteration > firstIteration ? iteration - firstIteration : 1;
    printf("ms per iteration: total %.3f, intersect %.3f, sort %.3f, shade %.3f, compact %.3f, denoise %.3f\n",
        guiData->msTotal / timed, guiData->msIntersect / timed, guiData->msSort / timed,
        guiData->msShade / timed, guiData->msCompact / timed, guiData->msDenoise / timed);
    printf("paths per bounce:");
    for (int count : guiData->pathCounts)
    {
        printf(" %d", count);
    }
    printf("\n");
}

void runCuda()
{
    if (camchanged)
    {
        iteration = resumeIteration;
        firstIteration = resumeIteration;
        if (resumeIteration == 0)
        {
            std::fill(renderState->image.begin(), renderState->image.end(), glm::vec3());
            std::fill(renderState->albedo.begin(), renderState->albedo.end(), glm::vec3());
            std::fill(renderState->normal.begin(), renderState->normal.end(), glm::vec3());
        }
        resumeIteration = 0;

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

        pathtraceFree();
        pathtraceInit(scene);
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

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
