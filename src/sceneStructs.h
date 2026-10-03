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

enum PrimitiveType
{
    PRIMITIVE_TRIANGLE,
    PRIMITIVE_GEOM
};

struct PrimitiveRef
{
    PrimitiveType type;
    int index;
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct DistantLight
{
    glm::vec3 toLight;
    glm::vec3 strength;
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
    glm::vec3 positions[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
    int materialId;
};

struct BVHNode
{
    glm::vec3 minCorner;
    glm::vec3 maxCorner;

    int leftChild = -1;
    int rightChild = -1;

    int firstPrimitive = 0;
    int primitiveCount = 0;
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
    int colorTextureId = -1;
    bool useCheckerboard = false;
    int bumpTextureId = -1;
    float bumpStrength = 0.02f;
    //wanted these lights to go onto my character but not the rest of the objects
    bool receiveAccentLights = true;
    float atmosphereStrength = 0.0f;
    float lightingMultiplier = 1.0f;
};

struct Texture
{
    int width;
    int height;
    int pixelOffset;
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
    glm::vec3 color; //throughput
    glm::vec3 radiance; //accumulated ligt for path
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
  glm::vec2 uv;
  glm::vec3 dpdu;
  glm::vec3 dpdv;
  glm::vec3 dndu;
  glm::vec3 dndv;
};
