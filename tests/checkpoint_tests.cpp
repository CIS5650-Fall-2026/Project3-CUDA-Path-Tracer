#include "render_checkpoint.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message); // Runs in Release too.
}
template<class F> static void rejects(F operation, const char* message) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
static checkpoint::Data fixture() {
    checkpoint::Data d;
    d.width = 2; d.height = 2; d.traceDepth = 8; d.completedSamples = 32;
    d.sceneIdentity = "{\"fixture\":true}"; d.buildIdentity = "codec-test-build";
    d.camera.position = {0, 0, 2}; d.camera.view = {0, 0, -1};
    d.camera.up = {0, 1, 0}; d.camera.right = {1, 0, 0};
    d.camera.fov = {45, 45}; d.camera.pixelLength = {0.1f, 0.1f};
    d.controller.zoom = 2; d.controller.theta = 1.57079632679f;
    d.controller.cameraPosition = {0, 0, 2};
    d.accumulation = {0, -0.0f, 0.125f, 1, 2, 3, 4.5f, 16, 0.25f, 100, 0, 1};
    return d;
}
int main(int argc, char** argv) {
    const auto suffix = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() / ("pathtracer-codec-" + std::to_string(suffix));
    try {
        auto d = fixture();
        if (argc == 3 && std::string(argv[1]) == "--fixture") {
            checkpoint::write(argv[2], d);
            std::cout << "Wrote production-codec fixture\n";
            return 0;
        }
        std::filesystem::create_directories(root);
        const auto path = root / "state.ptc";
        checkpoint::write(path, d);
        const auto decoded = checkpoint::read(path);
        require(decoded.completedSamples == 32 && decoded.width == 2 && decoded.sortMaterials,
            "Header roundtrip failed");
        require(decoded.accumulation.size() == d.accumulation.size() &&
            std::memcmp(decoded.accumulation.data(), d.accumulation.data(), d.accumulation.size() * 4) == 0,
            "RGB bits changed in roundtrip");
        require(decoded.camera.position == d.camera.position && decoded.controller.zoom == d.controller.zoom,
            "Camera/controller roundtrip failed");
        checkpoint::Expected expected{2, 2, 8, d.sceneIdentity, d.buildIdentity};
        checkpoint::requireCompatible(decoded, expected);
        auto wrong = expected; wrong.buildIdentity += "changed";
        rejects([&] { checkpoint::requireCompatible(decoded, wrong); }, "Accepted different build");
        wrong = expected; wrong.sceneIdentity += "changed";
        rejects([&] { checkpoint::requireCompatible(decoded, wrong); }, "Accepted different scene");
        wrong = expected; wrong.width = 3;
        rejects([&] { checkpoint::requireCompatible(decoded, wrong); }, "Accepted different dimensions");
        std::ifstream input(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
        input.close();
        bytes[bytes.size() - 9] ^= 1;
        const auto bad = root / "bad.ptc";
        { std::ofstream f(bad, std::ios::binary); f.write(bytes.data(), bytes.size()); }
        rejects([&] { checkpoint::read(bad); }, "Accepted corrupted RGB");
        { std::ofstream f(bad, std::ios::binary | std::ios::trunc); f.write(bytes.data(), 17); }
        rejects([&] { checkpoint::read(bad); }, "Accepted truncated file");
        auto nonfinite = d; nonfinite.accumulation[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { checkpoint::write(path, nonfinite); }, "Accepted NaN accumulation");
        require(checkpoint::read(path).completedSamples == 32, "Failed save damaged previous checkpoint");
        d.completedSamples = 128;
        checkpoint::write(path, d);
        require(checkpoint::read(path).completedSamples == 128, "Replacement save failed");
        std::filesystem::remove_all(root);
        std::cout << "PASS: production checkpoint codec, compatibility, corruption, replacement (CPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::error_code ignored; std::filesystem::remove_all(root, ignored);
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
