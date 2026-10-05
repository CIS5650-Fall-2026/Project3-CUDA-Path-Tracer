#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

enum MaterialType
{
    MATERIAL_DIFFUSE,
    MATERIAL_SPECULAR,
    MATERIAL_EMITTING,
    MATERIAL_REFRACTIVE
};

enum PrimitiveType
{
    PRIMITIVE_SPHERE,
    PRIMITIVE_CUBE,
    PRIMITIVE_TRIANGLE
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

struct Triangle
{
    glm::vec3 p0;
    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;
    int materialId;
};

struct Aabb
{
    glm::vec3 minimum;
    glm::vec3 maximum;
};

struct Primitive
{
    PrimitiveType type;
    int index;
    int materialId;
    Aabb bounds;
};

struct BvhNode
{
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    int leftmostChild;
    int rightChild;
    int primitiveCount;
};

struct Material
{
    MaterialType type;
    glm::vec3 color;
    struct
    {
        float roughness;
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
    float apertureRadius;
    float focusDistance;
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
    glm::vec3 throughput;  // better name than color
    int pixelIndex;
    int remainingBounces;
    float currentIor;
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
    bool outside;
};
