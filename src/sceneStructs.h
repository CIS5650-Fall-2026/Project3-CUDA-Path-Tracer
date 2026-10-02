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
    float time = 0.0f;
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
    glm::vec3 velocity = glm::vec3(0);
    glm::vec3 boundsMin = glm::vec3(0);
    glm::vec3 boundsMax = glm::vec3(0);
    int triangleStart = 0;
    int triangleCount = 0;
};

struct Triangle { glm::vec3 a, b, c; };

struct RenderOptions
{
    bool materialSorting = true;
    bool antialiasing = true;
    bool russianRoulette = true;
    bool refraction = true;
    bool depthOfField = true;
    bool directLighting = true;
    bool halton = true;
    bool motionBlur = true;
    bool compaction = true;
    bool sharedCompaction = true;
    bool meshCulling = true;
    bool postprocess = false;
    int rouletteDepth = 3;
    unsigned int seed = 1;
    float exposure = 0.0f;
    float gamma = 2.2f;
    float bloomStrength = 0.15f;
    float bloomThreshold = 1.0f;
    int bloomRadius = 6;
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
    float apertureRadius = 0.0f;
    float focalDistance = 10.0f;
    float shutterOpen = 0.0f;
    float shutterClose = 0.0f;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::vector<glm::vec3> displayImage;
    RenderOptions options;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    glm::vec3 radiance = glm::vec3(0);
    glm::vec3 previousPoint = glm::vec3(0);
    float previousPdf = 0.0f;
    float etaScale = 1.0f;
    bool previousDelta = true;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int geomId = -1;
  int triangleId = -1;
  glm::vec3 point;
  glm::vec3 outwardNormal;
  bool frontFace = true;
};
