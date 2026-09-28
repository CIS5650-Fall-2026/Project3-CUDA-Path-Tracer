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

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    int triangleStart = 0;
    int triangleCount = 0;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
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
    glm::vec3 a, b, c;
    glm::vec3 normal;
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
    float apertureRadius = 0.0f;
    float focalDistance = 1.0f;
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
    bool streamCompaction = true;
    bool materialSorting = false;
    bool antialiasing = true;
    bool meshCulling = true;
    bool directLighting = true;
    bool recordPathCounts = false;
    std::vector<int> activePaths;
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    bool useMis = false;
    float previousPdf = 0.0f;
    glm::vec3 previousHitPoint;
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
  int geomId;
  int triangleIndex;
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
