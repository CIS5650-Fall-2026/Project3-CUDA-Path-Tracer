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
    PLANE
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
    int lightid;
};

struct Light
{
    int geomid;
    float localPdf; // Assuming constant
};

struct Volume {
    glm::mat4 invTransform;
    glm::mat4 userInvTransform;
    float albedo;
    float extinctionMult; // For the entire volume, or should it be calculated along ray first?
    float g;
    int materialid;
};

struct VolumeData {
    Volume* volumes;
    int num_volumes;

    void* density;
    void* temperature;
    int bounceDepth;

    float gDecay;
    float densityDecay;
    float scatteringDecay;

    float temperatureScale;

    float emissionContrast;
    float emissionScale;

    float extinctionMult;
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

    float isVolume;
    float g1;
    float g2;
    float gBlend;
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
    float lastBsdfPdf;
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
  int pathIdx;
};
