#pragma once

#include "bsdf/bsdf_structs.cuh"
#include <cstdint>
#include <cuda_runtime.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

using MatId = uint8_t;
static constexpr uint8_t MAT_MISS = UINT8_MAX;
static constexpr uint8_t MAT_LIGHT = UINT8_MAX - 1;

enum GeomType { Sphere, Cube, Plane, Sdf };

struct Ray {
    glm::vec3 org;
    glm::vec3 dir;
};

struct Transform {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 matrix;
    glm::mat4 inverse;
    glm::mat4 inv_transpose;
};

// either a light or a material
enum class GeomRole : uint8_t { Material, Light };
struct Geom {
    GeomType type;
    GeomRole role;

    int id;

    Transform transform;
};

struct Emission {
    glm::vec3 color{1.f};
    float strength = 0.f;
    __host__ __device__ glm::vec3 radiance() const { return color * strength; }
};

// area lights (light -> geom)
struct Light {
    Emission emission;
    int geom_id;
};

struct Material {
    BsdfVariant bsdf;
    Emission emission;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    int pixel_index;
    int remaining_bounces;
    float prev_bsdf_pdf;
    bool prev_was_delta;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    glm::vec3 surface_normal;
    float t;
    int light_idx;
};

enum class BxdfFlag : uint8_t {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
};

// overload operators because BxdfFlag is strongly typed
__host__ __device__ constexpr BxdfFlag operator|(BxdfFlag a, BxdfFlag b) {
    return static_cast<BxdfFlag>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

__host__ __device__ constexpr BxdfFlag operator&(BxdfFlag a, BxdfFlag b) {
    return static_cast<BxdfFlag>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

__host__ __device__ constexpr BxdfFlag& operator|=(BxdfFlag& a, BxdfFlag b) {
    return a = a | b;
}

struct BsdfSample {
    glm::vec3 wi;
    float pdf;
    // cosine is baked in
    glm::vec3 f;
    BxdfFlag type;
};

struct BsdfEval {
    glm::vec3 f;
    float pdf;
};

struct LightSample {
    glm::vec3 wi;
    float dist;
    float pdf;
    // -1 for env
    int light_idx;
};

struct ShadowRay {
    Ray ray;
    int pixel_index;
    glm::vec3 contribution;
    float t_max;
};
