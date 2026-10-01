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
    MESH,
    MANDELBULB,
    MENGER
};

enum TextureType
{
    TEXTURE_NONE,
    TEXTURE_CHECKER,
    TEXTURE_MARBLE
};

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

// Leaf if count > 0, otherwise children are at first and first + 1
struct BVHNode
{
    glm::vec3 boundsMin;
    int first;
    glm::vec3 boundsMax;
    int count;
};

struct MeshData
{
    const Triangle* triangles;
    const BVHNode* nodes;
    bool useBoundsCulling;
    bool useBVH;
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
    glm::vec3 velocity;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // Mesh only
    int triangleStart;
    int triangleCount;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    int bvhRoot;
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

    int texture;
    glm::vec3 color2;
    float textureScale;
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

    float apertureRadius = 0.0f;
    float focalDistance = 1.0f;
};

struct Fog
{
    float density = 0.0f;
    glm::vec3 albedo = glm::vec3(1.0f);
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
    glm::vec3 radiance;
    int pixelIndex;
    int remainingBounces;
    float bsdfPdf;
    float time;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int geomId;
};
