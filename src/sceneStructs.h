#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

enum MeshAccel 
{ 
    ACCEL_NONE = 0, 
    ACCEL_AABB = 1, 
    ACCEL_BVH = 2 
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB {
    glm::vec3 minBound{ FLT_MAX,  FLT_MAX,  FLT_MAX };
    glm::vec3 maxBound{ -FLT_MAX, -FLT_MAX, -FLT_MAX };

    __host__ __device__ AABB() = default;

    __host__ __device__ AABB(const glm::vec3& minB, const glm::vec3& maxB)
        : minBound(minB), maxBound(maxB) {
    }

    // Expand bounding box to enclose a point
    __host__ __device__ void grow(const glm::vec3& p) {
        minBound = glm::min(minBound, p);
        maxBound = glm::max(maxBound, p);
    }

    // Expand bounding box to enclose another AABB
    __host__ __device__ void grow(const AABB& box) {
        minBound = glm::min(minBound, box.minBound);
        maxBound = glm::max(maxBound, box.maxBound);
    }
};

struct BVHNode {
    AABB bounds;
    int left = -1, right = -1;   // child node indices (internal nodes)
    int firstTri = 0, triCount = 0; // triCount > 0 => leaf
};

struct Triangle {
    glm::vec3 p0, p1, p2;
    glm::vec3 n0, n1, n2;
    glm::vec2 uv0, uv1, uv2;
    int materialId;
    AABB aabb;
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

    // Mesh specific attributes
    int triangleStartIdx = -1;
    int triangleCount = 0;
    AABB boundingBox; // Object-space AABB
    int bvhRoot = -1;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
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
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
};
