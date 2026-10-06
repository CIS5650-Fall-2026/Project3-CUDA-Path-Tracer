#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define STREAM_COMPACTION 1
#define MATERIAL_SORTING 0
#define BOUNDING_BOX 1

#define BVH 1
#define MAX_BVH_DEPTH 64

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Bounds
{
  glm::vec3 minBound;
  glm::vec3 maxBound;
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

    // mesh variables
    int numTriangles;
    Bounds bounds;

    int bvhRootIndex;
};

struct Triangle
{
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    int materialid;
};

struct LinearBVHNode {
  Bounds bounds;
  union {
    int primitivesOffset;   // leaf
    int secondChildOffset;  // interior
  };
  uint16_t nPrimitives;     // 0 = interior
  uint8_t axis;
  uint8_t pad[1];           // padding for 32 bytes
};

struct Material
{
    glm::vec3 color;

    float metallic;
    float roughness;
    float indexOfRefraction;
    float transmission;
    glm::vec3 emittance;
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

    float lensRadius;
    float focalDistance;
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
};
