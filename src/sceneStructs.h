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

enum MatType {
    DIFFUSE,
    SPECULAR,
    METALLICWORKFLOW,
    DIELECTRIC
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 lastHit;
    float distTraveled;
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

struct Material
{
    enum MatType type;
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;

    // point to texture in cudaTextureObjects_t array, -1 if none
    int texIdx = -1;
    int roughmapIdx = -1;

    float emittance;

    // for metallic workflow pbr
    float roughness;
    float metallic;

    // for diaeletric handling reflection & refraction 
    float refractionIndex;
    glm::vec3 absorption;
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
  glm::vec2 uv;
};

struct Triangle {
    int materialid;
    glm::vec3 positions[3];
    glm::vec3 normal;
    glm::vec3 centroid;
    glm::vec2 uv[3];
};

struct BVHNode
{
    glm::vec3 aabbMin, aabbMax;
    // leftFirst -> == index of left child if triCount = 0 (not a leaf) & == firstPrim if triCount > 0 (is a leaf) 
    int leftFirst, triCount;
};
