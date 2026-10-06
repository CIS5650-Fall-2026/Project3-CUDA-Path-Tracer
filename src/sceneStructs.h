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
    glm::vec3 v0, v1, v2;   
    glm::vec3 n0, n1, n2;   
    glm::vec2 t0, t1, t2;   // texture coordinates
    glm::vec3 tangent, bitangent;   // per-triangle, object space
};


// 32 bytes: two nodes per 64-byte cache line
struct BVHNode
{
    glm::vec3 bboxMin;
    int leftOrFirst;   // interior: left child index (right = left + 1); leaf: first triangle index
    glm::vec3 bboxMax;
    int triCount;      // 0 = interior, > 0 = leaf
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
    int triStart;
    int triCount;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int bvhRoot;
    float area;
};

struct TextureInfo
{
    int offset;   // first pixel in the packed texture array
    int width;
    int height;
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
    float roughness;   // GGX roughness, 0 = perfect mirror
    float metallic;    // 1 = metal (F0 = color), 0 = glossy coat over diffuse
    int texId;    // -1 = no texture
    int procTex;          // 0 = none, 1 = checker
    float procScale;      // checker cells per unit
    glm::vec3 procColor2; // second checker color
    int normalTexId;
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
    bool toneMap;
    float bloomStrength = 0.0f;   // 0 = off
    float bloomThreshold = 1.0f;  // linear radiance above which pixels glow
    int bloomRadius = 24;         // blur radius in pixels
    std::vector<glm::vec3> bloom; // host copy of the blurred glow, for saving
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float lastPdf;   // solid-angle pdf of last BSDF sample; <0 = delta/camera
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
  int geomId;
  glm::vec3 tangent, bitangent;
};
