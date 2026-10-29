#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

enum class GeomType
{
    SPHERE,
    CUBE,
    MESH  // one glTF primitive's triangles
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// One glTF primitive's triangles in the flat index array; indices are already offset, so no base vertex.
struct TriangleMesh
{
    int indexOffset;  // first triangle in the index array
    int triCount;
};

// Device pointers to the flat mesh arrays; read-only after upload.
struct MeshBuffers
{
    glm::vec3* positions;
    glm::vec3* normals;     // per vertex, unit length, object space
    glm::vec2* uvs;         // per vertex, TEXCOORD_0; (0, 0) for a primitive without one
    // Object space, xyz along increasing u; bitangent = w * cross(normal, tangent) as in glTF; zero without a normal map.
    glm::vec4* tangents;
    glm::ivec3* indices;    // per triangle, into the vertex arrays
    TriangleMesh* meshes;   // indexed by Geom::meshId
};

// 8-bit RGBA, rows top to bottom; glTF's uv (0, 0) is the top left, so fetches read it as stored.
struct TextureImage
{
    int width;
    int height;
    std::vector<unsigned char> rgba;  // width * height * 4
};

struct Texture
{
    int image;  // into Scene::textureImages
    cudaTextureAddressMode wrapU;
    cudaTextureAddressMode wrapV;
    cudaTextureFilterMode filter;
    bool srgb;
};

struct Geom
{
    GeomType type;
    int materialId;
    int meshId;  // MESH only: which TriangleMesh; -1 for the primitives
    // MESH only: -1 when transform mirrors, flipping the bitangent sign.
    float tangentSign;
    // Cycles-style light linking: bit k set receives group k; group 0 holds unlinked lights, so bit 0 is always set.
    unsigned int lightMask = 0xffffffffu;
    int lightGroup = 0;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum AlphaMode
{
    ALPHA_OPAQUE,
    ALPHA_MASK,  // a hit whose alpha is below alphaCutoff is no hit
    ALPHA_BLEND  // alpha is coverage: a hit counts with probability alpha
};

// glTF metallic-roughness with its extensions; defaults are glTF's default material. Texture slots are -1 for none.
struct Material
{
    glm::vec3 baseColor = glm::vec3(1.0f);
    float ior = 1.5f;                             // 0 means a Fresnel term of 1
    float alpha = 1.0f;
    AlphaMode alphaMode = ALPHA_OPAQUE;
    float alphaCutoff = 0.5f;
    float metallic = 1.0f;
    float roughness = 1.0f;
    float normalScale = 1.0f;                     // scales the normal map's x and y
    glm::vec3 emission = glm::vec3(0.0f);         // emissive factor * KHR_materials_emissive_strength
    float transmission = 0.0f;
    float specularFactor = 1.0f;
    glm::vec3 specularColorFactor = glm::vec3(1.0f);
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.0f;
    // Per unit of world distance: -ln(attenuationColor) / attenuationDistance.
    glm::vec3 absorption = glm::vec3(0.0f);
    int baseColorTexture = -1;                    // sRGB rgb, linear alpha
    int metallicRoughnessTexture = -1;            // G roughness, B metallic
    int normalTexture = -1;                       // tangent space, +Y up
    int emissiveTexture = -1;                     // sRGB
    int transmissionTexture = -1;                 // R
};

// The pinhole stays as the cmp reference, the thin lens as the baseline against the real lens.
enum CameraType
{
    CAMERA_PINHOLE,
    CAMERA_THIN_LENS,
    // Traced surface by surface from the film out.
    CAMERA_REAL_LENS,
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 pixelLength;
    // right is -cross(view, up), flipping the image left to right.
    bool mirrored;
    CameraType type;
    // Scene units.
    float apertureRadius;
    float focusDistance;
    // Sizes the real lens (in mm) against the scene.
    float unitsPerMeter;
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
    // Material the path is inside, -1 in air.
    int medium;
    // Solid-angle pdf over the rough lobes; 0 for a camera ray or smooth lobe, whose hit emission counts without MIS.
    float pdf;
    // Geom::lightMask of the surface the ray left; all ones for a camera ray.
    unsigned int receiverMask;
};

struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;  // (0, 0) on spheres and cubes
  // World space, not unit length; w already includes a mirroring transform. Zero without a tangent.
  glm::vec4 tangent;
  int materialId;
  bool outside;
  // |cos| to the flat-triangle normal, matching NEE's light pdf; written by the OptiX hit programs only.
  float cosGeometric;
  unsigned int lightMask;
  int lightGroup;
};

// cross(e1, e2) is the front-face normal, so a mirroring transform swaps corners 1 and 2.
struct LightTriangle
{
    glm::vec3 p0;
    glm::vec3 e1;  // p1 - p0
    glm::vec3 e2;  // p2 - p0
    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;
    int materialId;
    int group;  // the Geom's lightGroup
};

enum PunctualLightType
{
    LIGHT_POINT,
    LIGHT_SPOT,
    LIGHT_DISTANT
};

// A light without a surface; only NEE reaches it, without a MIS weight.
struct PunctualLight
{
    PunctualLightType type;
    glm::vec3 position;   // LIGHT_POINT, LIGHT_SPOT: where it is; LIGHT_DISTANT: unit direction toward it
    // W/sr for point and spot (on axis), W/m^2 irradiance facing it for distant.
    glm::vec3 intensity;
    // LIGHT_SPOT only; cones as half-angle cosines, cosInner == cosOuter is a hard edge.
    glm::vec3 direction;
    float cosOuter;
    float cosInner;
    int group;            // light linking group, 0 for an unlinked light
};

struct ShadowRay
{
    glm::vec3 origin;
    glm::vec3 direction;     // unit length
    float tMax;              // just short of the point on the light
    glm::vec3 contribution;  // MIS weight included
    int pixelIndex;
    unsigned int alphaSeed;  // seeds the ALPHA_BLEND test
};
