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

// One triangle of a loaded mesh, in the mesh's object space. Normals are
// per-vertex so that smooth shading works; a file without normals gets the
// face normal copied into all three, which makes the same code path flat shade.
struct Triangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
};

// One node of a mesh's bounding volume hierarchy, 32 bytes so the GPU can
// fetch it as two 16-byte loads. The two children of an internal node are
// always stored next to each other, so only the left one needs an index.
struct BVHNode
{
    glm::vec3 bboxMin;
    int leftOrFirst;   // internal: index of the left child (right = left + 1)
                       // leaf: first triangle in the scene's triangle buffer
    glm::vec3 bboxMax;
    int triCount;      // 0 for an internal node, > 0 for a leaf
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

    // MESH only: the half-open range [triangleStart, triangleStart + triangleCount)
    // into the scene's flat triangle buffer, plus the object-space bounding box
    // used to reject rays before walking that range.
    int triangleStart;
    int triangleCount;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;

    // MESH only: root of this mesh's BVH in the scene's node buffer, or -1 for
    // a mesh with no triangles.
    int bvhRoot;
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

    // Thin lens. A lens radius of 0 is a pinhole camera, where everything is in
    // focus; otherwise points at focalDistance along the view are sharp and the
    // rest blur in proportion to the lens radius.
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
  bool outside;
};
