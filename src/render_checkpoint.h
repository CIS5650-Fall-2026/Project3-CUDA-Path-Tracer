#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// no CUDA/GLM dependencies. serialized fields have explicit
// sizes and little-endian encoding...no C++ object layout is written to disk.
namespace checkpoint {

inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::uint32_t kRendererVersion = 1;

struct Camera {
    std::array<float, 3> position{}, lookAt{}, view{}, up{}, right{};
    std::array<float, 2> fov{}, pixelLength{};
};

struct Controller {
    float zoom = 0.0f, theta = 0.0f, phi = 0.0f;
    std::array<float, 3> originalLookAt{}, cameraPosition{};
};

struct Data {
    std::uint32_t width = 0, height = 0, traceDepth = 0, completedSamples = 0;
    bool sortMaterials = true;
    std::string sceneIdentity; // Full canonical source content, not its pathname.
    std::string buildIdentity; // Rendering source/compiler/settings signature.
    Camera camera;
    Controller controller;
    std::vector<float> accumulation; // unnormalized interleaved RGB binary32.
};

struct Expected {
    std::uint32_t width, height, traceDepth;
    std::string sceneIdentity, buildIdentity;
};

void requireCompatible(const Data& saved, const Expected& current);
Data read(const std::filesystem::path& path);
// 1 writer per destination. temp file is in the destination directory.
// Throws on failure...never delete the last completed checkpoint first.
void write(const std::filesystem::path& path, const Data& data);

}
