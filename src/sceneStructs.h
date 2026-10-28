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
    // MESH only: slice of Scene::triangles plus its world-space bounding box
    int triStart;
    int triCount;
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    int bvhRoot;   // index into Scene::bvhNodes of the root of this mesh's hierarchy
};

// Procedural patterns a material can compute from uv instead of reading an image
enum ProceduralType
{
    PROC_NONE,
    PROC_TILES
};

// How a surface scatters light, one per material
enum MaterialType
{
    DIFFUSE,
    EMISSIVE,
    SPECULAR,
    DIELECTRIC
};

struct Material
{
    MaterialType type;
    glm::vec3 color;      // base color; an image or a pattern below replaces it per hit
    float emittance;      // EMISSIVE only
    float exponent;       // SPECULAR only: Phong exponent of the lobe, 0 is a perfect mirror
    float ior;            // DIELECTRIC only: index of refraction

    // where the base color comes from when it is not the constant above
    struct
    {
        int image;        // index into the texture table, -1 when the material has no image
        int procedural;   // a ProceduralType, PROC_NONE when the color is not computed
        float bump;       // how hard the pattern's height tilts the normal, 0 turns bump off
    } surface;

    // parameters of PROC_TILES
    struct
    {
        glm::vec2 count;      // tiles across one unit of uv, along u and along v
        glm::vec2 grout;      // grout thickness as a fraction of one tile, per axis
        glm::vec3 groutColor; // color of the lines between tiles
    } tiles;
};

// One loaded image, stored as a slice of the shared texel array
struct TextureInfo
{
    int offset;   // index of this image's first texel
    int width;
    int height;
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
    float lensRadius;      // 0 keeps the pinhole camera, where everything is sharp
    float focalDistance;   // distance along the view direction to the plane that stays sharp
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
  glm::vec2 uv;
  glm::vec3 tangent;   // world-space direction in which u increases across the surface
};


// Stored in world space; fixed arrays only because this is memcpy'd to the GPU
struct Triangle
{
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
};

// One node of a mesh's bounding volume hierarchy
// A leaf owns a slice of Scene::triangles, an interior node owns two children
struct BVHNode
{
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    int left;       // index into Scene::bvhNodes, -1 on a leaf
    int right;
    int triStart;
    int triCount;   // 0 on an interior node
};
