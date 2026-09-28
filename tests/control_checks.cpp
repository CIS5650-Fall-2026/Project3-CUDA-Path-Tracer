// Include the renderer so these tests can use its controls without changing main.cpp.
#define main rendererMain
#include "../src/main.cpp"
#undef main

#include <filesystem>
#include <stdexcept>

GLFWkeyfun sendKey;
GLFWmousebuttonfun sendButton;
GLFWcursorposfun moveMouse;

void check(bool passed, const char* message) {
    if (!passed) {
        throw std::runtime_error(message);
    }
    std::cout << "PASS: " << message << std::endl;
}

bool closeEnough(float a, float b) {
    return std::abs(a - b) < 0.0001f;
}

bool samePosition(glm::vec3 a, glm::vec3 b) {
    return glm::length(a - b) < 0.0001f;
}

bool setup(const char* sceneFile) {
    scene = new Scene(sceneFile);
    renderState = &scene->state;
    width = renderState->camera.resolution.x;
    height = renderState->camera.resolution.y;
    ogLookAt = renderState->camera.lookAt;
    guiData = new GuiDataContainer();

    if (!init()) {
        return false;
    }
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW returns the old callback when we set a new one.
    // Save each one, then put it back so ImGui stays connected.
    sendKey = glfwSetKeyCallback(window, nullptr);
    sendButton = glfwSetMouseButtonCallback(window, nullptr);
    moveMouse = glfwSetCursorPosCallback(window, nullptr);
    glfwSetKeyCallback(window, sendKey);
    glfwSetMouseButtonCallback(window, sendButton);
    glfwSetCursorPosCallback(window, moveMouse);
    return true;
}

void resetCamera() {
    renderState->camera.lookAt = ogLookAt;
    zoom = 10.5f;
    phi = 0.0f;
    theta = PI / 2.0f;
    leftMousePressed = false;
    rightMousePressed = false;
    middleMousePressed = false;
    mouseOverImGuiWinow = false;
    camchanged = true;
    runCuda();
}

void testZoom() {
    resetCamera();
    runCuda();
    float oldZoom = zoom;

    sendButton(window, GLFW_MOUSE_BUTTON_RIGHT, GLFW_PRESS, 0);
    double startX = lastX;
    double startY = lastY;

    // Move only vertically. This used to get ignored because x did not change.
    moveMouse(window, startX, startY + height / 10.0);
    check(closeEnough(zoom, oldZoom + 0.1f), "vertical drag changes zoom");

    runCuda();
    Camera& camera = renderState->camera;
    float distance = glm::length(camera.position - camera.lookAt);
    check(iteration == 1 && closeEnough(distance, zoom), "zoom moves the camera and resets samples");

    moveMouse(window, startX, startY - height * 20.0);
    check(closeEnough(zoom, 0.1f), "zoom cannot go below 0.1");
    sendButton(window, GLFW_MOUSE_BUTTON_RIGHT, GLFW_RELEASE, 0);
}

void testPan() {
    resetCamera();
    runCuda();
    Camera& camera = renderState->camera;
    glm::vec3 oldTarget = camera.lookAt;
    glm::vec3 oldPosition = camera.position;

    sendButton(window, GLFW_MOUSE_BUTTON_MIDDLE, GLFW_PRESS, 0);
    double startX = lastX;
    double startY = lastY;

    // At this camera angle, 80 pixels moves the target 0.8 units.
    moveMouse(window, startX + 80, startY);
    check(closeEnough(camera.lookAt.x, oldTarget.x - 0.8f), "horizontal drag pans sideways");

    moveMouse(window, startX + 80, startY + 80);
    bool movedForward = closeEnough(camera.lookAt.z, oldTarget.z - 0.8f);
    bool sameHeight = closeEnough(camera.lookAt.y, oldTarget.y);
    check(movedForward && sameHeight, "vertical drag pans along the ground");

    glm::vec3 panOffset = camera.lookAt - oldTarget;
    runCuda();
    glm::vec3 expectedPosition = oldPosition + panOffset;
    check(iteration == 1 && samePosition(camera.position, expectedPosition),
        "pan moves the camera with the target and resets samples");
    sendButton(window, GLFW_MOUSE_BUTTON_MIDDLE, GLFW_RELEASE, 0);
}

void testDraggingOverUI() {
    resetCamera();
    sendButton(window, GLFW_MOUSE_BUTTON_MIDDLE, GLFW_PRESS, 0);

    // Start outside the panel, then release over it.
    mouseOverImGuiWinow = true;
    sendButton(window, GLFW_MOUSE_BUTTON_MIDDLE, GLFW_RELEASE, 0);
    moveMouse(window, lastX + 20, lastY + 20);
    check(!middleMousePressed && !camchanged, "releasing over the panel stops the drag");

    sendButton(window, GLFW_MOUSE_BUTTON_RIGHT, GLFW_PRESS, 0);
    check(!rightMousePressed, "clicking the panel does not start a camera drag");
    mouseOverImGuiWinow = false;
}

void testSpace() {
    resetCamera();
    Camera& camera = renderState->camera;
    camera.lookAt += glm::vec3(1.0f, 0.0f, 1.0f);
    camchanged = true;
    runCuda();
    runCuda();
    glm::vec3 movedTarget = camera.lookAt;
    float oldZoom = zoom;

    sendKey(window, GLFW_KEY_SPACE, 0, GLFW_RELEASE, 0);
    check(samePosition(camera.lookAt, movedTarget), "releasing Space leaves the target alone");

    sendKey(window, GLFW_KEY_SPACE, 0, GLFW_PRESS, 0);
    check(samePosition(camera.lookAt, ogLookAt) && closeEnough(zoom, oldZoom),
        "Space restores the target without changing zoom");

    runCuda();
    check(iteration == 1, "Space resets samples");
}

std::string imagePath() {
    return renderState->imageName + "." + startTimeString + "."
        + std::to_string(iteration) + "samp.png";
}

void testSave(const std::filesystem::path& outputFolder) {
    resetCamera();
    renderState->imageName = (outputFolder / "save").string();
    std::string filename = imagePath();
    check(!std::filesystem::exists(filename), "no old save image is present");

    sendKey(window, GLFW_KEY_S, 0, GLFW_RELEASE, 0);
    check(!std::filesystem::exists(filename), "releasing S does not save");

    sendKey(window, GLFW_KEY_S, 0, GLFW_PRESS, 0);
    bool saved = std::filesystem::exists(filename);
    check(saved && std::filesystem::file_size(filename) > 100, "S writes an image");
    check(!glfwWindowShouldClose(window), "S keeps the window open");
}

void testEscape(const std::filesystem::path& outputFolder) {
    resetCamera();
    renderState->imageName = (outputFolder / "escape").string();

    sendKey(window, GLFW_KEY_ESCAPE, 0, GLFW_PRESS, 0);
    check(glfwWindowShouldClose(window), "Escape requests the window to close");
    check(std::filesystem::exists(imagePath()), "Escape also saves an image");
}

void cleanup() {
    pathtraceFree();
    cleanupCuda();
    // cleanupCuda also runs at exit. These buffers have already been freed.
    pbo = 0;
    displayImage = 0;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    delete guiData;
    delete scene;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: control_checks SCENEFILE.json\n";
        return 1;
    }
    if (!setup(argv[1])) {
        std::cerr << "Could not open the renderer.\n";
        return 1;
    }

    int result = 0;
    try {
        startTimeString = currentTimeString();
        std::filesystem::path outputFolder = std::filesystem::path("build")
            / ("control-check-" + startTimeString);
        std::filesystem::create_directories(outputFolder);

        resetCamera();
        check(iteration == 1, "first frame renders");
        testZoom();
        testPan();
        testDraggingOverUI();
        testSpace();
        testSave(outputFolder);
        testEscape(outputFolder);
        std::cout << "All checks passed. Images saved in " << outputFolder << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        result = 1;
    }
    cleanup();
    return result;
}
