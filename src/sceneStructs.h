#pragma once

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
    glm::vec3 origin{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
};

struct Geom
{
    enum GeomType type = SPHERE;
    int materialid = -1;
    glm::vec3 translation{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::mat4 transform{1.0f};
    glm::mat4 inverseTransform{1.0f};
    glm::mat4 invTranspose{1.0f};

    glm::vec3 vertices[3]{};
    glm::vec3 normals[3]{};
    glm::vec2 uvs[3]{};
    bool hasNormals = false;
    bool hasUVs = false;
};

struct Material
{
    glm::vec3 color{1.0f};
    struct
    {
        float exponent = 0.0f;
        glm::vec3 color{1.0f};
    } specular;
    float hasReflective = 0.0f;
    float hasRefractive = 0.0f;
    float indexOfRefraction = 1.0f;
    float emittance = 0.0f;
    int textureId = -1;
    int bumpTextureId = -1;
    int procedural = 0;
    float bumpStrength = 1.0f;
    float checkerScale = 8.0f;
};

struct Camera
{
    glm::ivec2 resolution{0};
    glm::vec3 position{0.0f};
    glm::vec3 lookAt{0.0f, 0.0f, -1.0f};
    glm::vec3 view{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    glm::vec3 right{1.0f, 0.0f, 0.0f};
    glm::vec2 fov{0.0f};
    glm::vec2 pixelLength{0.0f};
    float aperture = 0.0f;
    float focusDistance = 1.0f;
};

struct RenderState
{
    Camera camera{};
    unsigned int iterations = 0;
    int traceDepth = 0;
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
