#pragma once

#include <glm/glm.hpp>

#include <optional>
#include <string>

class Scene;

// A glTF perspective camera in world space, looking down its node's -Z with +Y up.
struct GltfCamera
{
    glm::vec3 position;
    glm::vec3 view;     // unit length
    glm::vec3 up;       // unit length
    bool mirrored;      // the transform flips handedness
    float yfov;         // full vertical field of view, radians
    float aspectRatio;  // width / height, 0 when the file leaves it to the viewport
};

// PBRT render settings from the root extras' pbrt.render (github.com/ErfanMo77/gltf-research-scenes); 0 = not in the file.
struct GltfRenderHints
{
    int width = 0;
    int height = 0;
    int maxDepth = 0;
};

// A research scene's PBRT infinite light from the root extras' pbrt.light_sources.
// PBRT maps are equal-area octahedral; tools/convert_envmaps.py writes <map>.latlong.exr next to each, rotation baked in.
struct GltfEnvironment
{
    std::string latlongFile;  // empty for a constant light
    std::string pbrtFile;
    glm::vec3 radiance;       // radiance_rgb of a constant light, 1 for a map
};

// What a file holds besides the geometry and materials that went into the scene.
struct GltfInfo
{
    // World space; min > max when nothing was added.
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    // First perspective camera, depth first from the default scene's roots.
    std::optional<GltfCamera> camera;
    GltfRenderHints render;
    std::optional<GltfEnvironment> environment;
};

// Appends a glTF file's default scene (meshes, materials, punctual lights) to the scene; false when it cannot be loaded.
// materialOverride >= 0 replaces every primitive's material. Only triangle lists with float positions load.
// Without info, PBRT infinite and distant lights are reported as ignored.
bool loadGltf(const std::string& path, const glm::mat4& sceneTransform, int materialOverride, Scene& scene,
    GltfInfo* info = nullptr);
