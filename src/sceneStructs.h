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
    MANDELBULB,
    MENGER,
    MESH
};

enum TextureType
{
    NO_TEXTURE,
    CHECKER,
    MARBLE
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
    glm::vec3 velocity;
    int triangleStart;
    int triangleCount;
    int bvhRoot;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
};

struct Triangle
{
    glm::vec3 positions[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
    int materialId;
};

struct BVHNode
{
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    int left;
    int right;
    int triangleStart;
    int triangleCount;
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
    float textureScale = 1.0f;
    int colorTexture = -1;
    int bumpTexture = -1;
    int emissiveTexture = -1;
    float bumpScale;
    bool normalMap;
    float scatterDistance;
};

struct Texture
{
    int width;
    int height;
    int offset;
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
    std::vector<glm::vec3> albedo;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec3> denoised;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float time;
    glm::vec3 mediumColor;
    float mediumDistance;
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
  glm::vec3 tangent;
};
