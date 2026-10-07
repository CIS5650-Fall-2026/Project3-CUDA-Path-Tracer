#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum GeomType
{
    SPHERE,
    CUBE
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
};

struct Triangle {
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
    glm::vec3 t0, t1, t2;
    float tangentSign0, tangentSign1, tangentSign2;
    glm::vec2 uv0, uv1, uv2;
    glm::vec3 centroid;
    int materialid;
};

enum AlphaMode : uint8_t
{
    ALPHA_MODE_OPAQUE = 0,
    ALPHA_MODE_MASK = 1,
    ALPHA_MODE_BLEND = 2
};

struct Material
{
    glm::vec3 color;
    float alpha;
    uint8_t alphaMode;
    float alphaCutoff;

    float metalic_factor;
    float roughness_factor;

    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;

    uint8_t is_emissive;
    glm::vec3 emissive_factor; // Emission color

    uint8_t double_sided; // For if we cull back faces

    float transmission_factor;
    float indexOfRefraction;

    int baseColorTexId;
    int metallicRoughnessTexId;
    int emissiveTexId;
    int normalTexId;

    float thicknessFactor;
    glm::vec3 attenuationColor;
    float attenuationDistance;

    Material() : metalic_factor(0.0f), roughness_factor(1.0f),
        is_emissive(0), double_sided(0), transmission_factor(0.0f),
        indexOfRefraction(1.0f), emissiveTexId(-1), normalTexId(-1),
        baseColorTexId(-1), metallicRoughnessTexId(-1),
        alphaMode(ALPHA_MODE_OPAQUE), alphaCutoff(0.5f),
        thicknessFactor(0.0f), attenuationColor(1.0f),
        attenuationDistance(3.402823466e+38f) {}
};

struct Texture {
    int width;
    int height;
    int channels;
    bool isHdr;
    std::vector<uchar4> pixels;
    std::vector<glm::vec4> hdrPixels;
};

struct DeviceTexture {
    int width;
    int height;
    int channels;
    int isHdr;
    uchar4 *pixels;
    glm::vec4 *hdrPixels;
};

enum SceneLightType : uint8_t
{
    SCENE_LIGHT_TRIANGLE = 0,
    SCENE_LIGHT_GEOM = 1
};

struct SceneLight
{
    uint8_t type;
    int id;
    float area;
};

struct LightSample
{
    SceneLight light;
    int lightIndex;
    float lightPickPdf;
    bool valid;
};

struct LightPointSample
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 barycentric;
    int materialId;
    float pdfArea;
    bool valid;
};

struct NeeSample
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 wi;
    glm::vec3 Li;
    float distance;
    float pdfArea;
    float pdfDirectional;
    float pdfLight;
    int lightIndex;
    int materialId;
    bool valid;
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

    glm::vec3 radiance;
    glm::vec3 throughput;

    int pixelIndex;
    int remainingBounces;

    float etaScale;
    int mediumMaterialIndex;

    float lastBsdfPdf;
    bool lastBounceWasSpecular;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 surfaceTangent;
  float tangentSign;
  int materialId;
  int geomId;
  int triangleId;
  uint8_t outside;
  glm::vec2 uv;
};
