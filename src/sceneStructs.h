#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include "generated/metal_presets.h"
#include "generated/transmission_presets.h"
#include <algorithm>
#include <cmath>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum class MaterialType : int
{
    Standard,
    Metal,
    Transmission
};

struct Material
{
    MaterialType type = MaterialType::Standard;

    // Linear RGB coefficients, not sampling probabilities.
    glm::vec3 Kd = glm::vec3(0.25f);
    glm::vec3 Ks = glm::vec3(0.25f);

    // Trowbridge-Reitz roughness-to-alpha mapping from pbrt (min actual roughness is 0.001)
    static float roughnessToAlpha(float r)
    {
        const float x = std::log(std::max(r, 0.001f));

        return 1.62142f
            + 0.819955f * x
            + 0.1734f * x * x
            + 0.0171201f * x * x * x
            + 0.000640711f * x * x * x * x;
    }

    // User-facing roughness in [0, 1].
    float roughness = 0.1f;

    // Initialize even when the scene omits roughness.
    float alpha = roughnessToAlpha(roughness);

    // CPU-side setter; r is valdated by the scene parser.
    void setRoughness(float r)
    {
        roughness = r;
        alpha = roughnessToAlpha(r);
    }

    // Standard materials only; air, IOR = 1.
    float ior = 1.5f;
    float diffuseSigmaDegrees = 0.0f; // Oren-Nayar: 0 (lambertian), 90.

    // Nonzero emitted RGB means a terminal light.
    struct Emission {
        glm::vec3 color = glm::vec3(0.0f);
        float emittance = 1.0f;
    } emission;

    struct Metal {
        // -1 means unused/unselected, not a renderable metal preset.
        int presetID = -1;
        glm::vec3 etaT = glm::vec3(1.0f);
        glm::vec3 k = glm::vec3(0.0f);

        // CPU only. Upload the material after selecting a preset.
        void setPreset(int id)
        {
            constexpr int count = static_cast<int>(
                sizeof(metalIORPresets) / sizeof(metalIORPresets[0]));

            if (id < 0 || id >= count) {
                throw std::out_of_range("Invalid metalPresetID");
            }

            presetID = id;
            etaT = metalIORPresets[id].eta;
            k = metalIORPresets[id].k;
        }
    } metal;

    struct Transmission {
        int presetID = -1; // Unselected, not a renderable transmission material.
        float ior = 1.5f;

        void setPreset(int id)
        {
            constexpr int count = static_cast<int>(
                sizeof(transmissionIORPresets) / sizeof(transmissionIORPresets[0]));
            if (id < 0 || id >= count) {
                throw std::out_of_range("Invalid transmissionPresetID");
            }
            presetID = id;
            ior = transmissionIORPresets[id].ior;
        }
    } transmission;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    bool sortMaterials = true;
    bool compactPaths = true;
    bool antialiasing = true;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    // Throughput while active; final emitted-light contribution when finished.
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t = -1.0f;
    glm::vec3 surfaceNormal = glm::vec3(0.0f); // Outward geometric normal.
    int materialId = -1;
    glm::vec3 position = glm::vec3(0.0f);      // Unoffset world-space hit.
};
