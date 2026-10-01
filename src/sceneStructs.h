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
    TORUS,
    MENGER
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
    float time;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::vec3 motion;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum ProceduralTextureType
{
    TEXTURE_NONE = 0,
    TEXTURE_CHECKER = 1,
    TEXTURE_STRIPES = 2,
    TEXTURE_FILE = 3
};

struct TextureData
{
    int width;
    int height;
    std::vector<glm::vec3> pixels;
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
    int textureType;
    glm::vec3 textureColor;
    float textureScale;

    int textureWidth;
    int textureHeight;
    int textureIndex;
    int textureOffset;

    int bumpWidth;
    int bumpHeight;
    int bumpOffset;
    float bumpStrength;
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
    bool allowEmission;
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
};

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    glm::vec3 normal;

    int materialId;
};

struct BVHNode
{
    glm::vec3 minBounds;
    glm::vec3 maxBounds;

    int leftChild;
    int rightChild;

    int triangleStart;
    int triangleCount;
};