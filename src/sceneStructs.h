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

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
};

// Children of an interior node are 8 consecutive nodes starting at firstChild,
// ordered by octant bits (x = 1, y = 2, z = 4).
struct OctNode
{
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int firstChild; // -1 for a leaf o.t. offset into 8 children nodes
    int triStart;   // leaf only offset into the triangle index buffer
    int triCount;   // leaf only
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

    //Mesh only
    int triangleStart;
    int triangleCount;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int octreeRoot;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective; // probability of a mirror bounce instead of a diffuse one
    float roughness;     // 0 = perfect mirror, 1 = reflection as blurry as diffuse
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

//for passing into stream compaction to remove dead paths
struct PathAlive
{
    __host__ __device__ bool operator()(const PathSegment& p)
    {
        return p.remainingBounces > 0;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
