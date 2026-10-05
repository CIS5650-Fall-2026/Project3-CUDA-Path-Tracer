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
    TRIANGLE
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
    // for triangles only
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 norm0 = glm::vec3(0.0f);
    glm::vec3 norm1 = glm::vec3(0.0f);
    glm::vec3 norm2 = glm::vec3(0.0f);
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

struct Bounds
{
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
};

struct BVHPrimitive
{
    Geom boundingBox;
    Bounds boundingCorners;
    int leafGeomIndex; // actual triangle for this primitive
};

struct BVHNode
{
    Bounds boundingCorners;
    Geom boundingBox;
    int bvhNodeChildIndex_First = -1; 
    int bvhNodeChildIndex_Second = -1;
    int numPrims = -1;
    int firstPrimOffset = -1;
    int dim = -1; //axis it was split on
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
    glm::vec3 throughput;
    float pdf;
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
  glm::vec3 intersectionPoint;
  int materialId;
};
