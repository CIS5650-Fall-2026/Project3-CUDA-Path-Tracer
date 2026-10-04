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
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    size_t triangleStart; 
    size_t triangleCount;
    int bvhRoot;
    bool hasUVs;
    bool visibleInGlass;
};

struct BVHNode
{
    glm::vec3 boundsMin, boundsMax;
    size_t leftChild, rightChild;
    size_t triangleStart, triangleCount;
};

struct Triangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
    glm::vec2 uv0, uv1, uv2;
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
    float roughness;
    int colorTex;              // colorTex = -1 means no texture. Index in scene texture list
    int bumpTex;
    float texScale;
    float bumpStrength;
    glm::vec2 bumpTexelSize;   // 1 / bump texture size
    bool triplanar;            // project along object X/Y/Z instead of using mesh UVs
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
    float lensRadius;      // aperture radius in world units. We calculate it with focalLength / (2.f * fstop). If 0, we have a pinhole camera.
    float focalDistance;   // world unit distance along the view direction to the plane of focus
    float focalLength;     // in world units
    float focalLengthMm;   // in mm
    float fstop;           // FSTOP from the scene file (0 means no DOF)
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
    bool specularBounce;   // whether or not last scatter was a delta material (specular or glass) or camera ray. Used for MIS
    float bsdfPdf;         // solid-angle pdf of the direction just BSDF-sampled (only meaningful when !specularBounce)
    bool afterGlass;
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
  int geomId;
  int triId;
};
