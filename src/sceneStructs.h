#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define BVH_LEAF_SIZE 4
#define BVH_STACK_SIZE 32
#define USE_BVH 1
#define BOUNDING_VOLUME_CULLING 1

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct BVHNode {
    int left; // child node index (leaf is -1)
    int right;
    int triStart;
    int numTris;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
};

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 tangent;
    glm::vec2 uv0;
    glm::vec2 uv1;
	glm::vec2 uv2;
};

struct Texture
{
    int width;
    int height;
    // offset is the position in the texel buffer/array.
    // because we want contiguous memory, we want to avoid having a pointer per texture.
    int offset;
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
    // triangle specific data
    int triStart;
    int numTris;
    glm::vec3 bboxMin;
	glm::vec3 bboxMax;
    // BVH
    int bvhRoot;
    // MIS
    float area;
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
    
    int albedoTexId;
    int bumpTexId;
    float bumpStrength;

    // 0: no noise
    // 1: fbm perlin
    // 2: voronoi
    // 3: voronoi with fbm input
    int noiseId;
    float noiseScale;
    // at 0, pure voronoi
    float warpStrength;
    float warpFreq;

    // ** For subsurface scattering
    // 0: none
    // increasing => more opaque
    float sssDensity;
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
    // a thin lens camera
    // at 0 radius, it's a pinhole and there is no blurring
    float lensRadius;
    // this is the distance along the view dir to the "sharp" plane
    float focalDistance;
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
    glm::vec3 radiance;
    float lastPdf;
    int pixelIndex;
    int remainingBounces;
    // ** For subsurface scattering
    // -1 if path NOT in a sss object
    int mediumMaterial;
    // light weighting requires "real" distance
    // this is for transparent cases
    float passDist;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 tangent;
  glm::vec2 uv;
  int materialId;
  int geomId;
  bool outside;
};
