// Struct layouts and constants shared by C++ and GLSL.
#ifndef GPU_SHARED_H
#define GPU_SHARED_H

#ifdef __cplusplus

#include <cstdint>
#include <glm/glm.hpp>

#define GPU_NAMESPACE_BEGIN namespace gpu {
#define GPU_NAMESPACE_END }
#define GPU_CONST static constexpr

// Alias C++/GLM types to the matching GLSL types for compatibility
GPU_NAMESPACE_BEGIN
using uint = uint32_t;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using uvec2 = glm::uvec2;
GPU_NAMESPACE_END

#else // GLSL

// Enable device buffer addressing, 64-bit integer types and scalar block layout
// (GLSL types aligned to their component size, matching C++'s data layout).
// Scalar block layout avoids standard GLSL padding, so a vec3 will align to 4 bytes.
// 64-bit integer lets us use int64_t and uint64_t in both GLSL and C++, which simplifies
// buffer device address management.
// 
// Note that scalar layout only applies to blocks declared with layout(scalar).
// #extension must precede declarations, so other shaders will need to include
// this header first.
//
// Also critical: every layout(...) must include "scalar", and if its a buffer
// being used with buffer device address then we additionally need both
// "buffer_reference" and "buffer_reference_align" 
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

#define GPU_NAMESPACE_BEGIN
#define GPU_NAMESPACE_END
#define GPU_CONST const

#endif

GPU_NAMESPACE_BEGIN

// Constants for descriptor set numbers and binding indices
GPU_CONST uint kRendererDescriptorSet = 0;
GPU_CONST uint kTLASBinding = 0;
GPU_CONST uint kAccumImageBinding = 1;
GPU_CONST uint kDisplayImageBinding = 2;

// Constants for material types
GPU_CONST uint kDiffuseRefl = 1;
GPU_CONST uint kSpecRefl = 2;
GPU_CONST uint kSpecTrans = 3;
GPU_CONST uint kSpecGlass = 4;

// Constants for flags
GPU_CONST uint kFlagDepthOfField = 0x1u;

// Misc.
GPU_CONST uint kInstanceMaskAllBitsSet = 0xFFu;

struct Vertex
{
    vec3 position;
    vec3 normal;
    vec2 uv;
}; 

// We'll have one GeometryInfo for each glTF primitive of each mesh instance
// Rows repeat per instance (not per mesh) so that instances sharing a BLAS
// can have different materials.
// 
// Index values in the index buffer are primitive-local (e.g. they start at 0
// for every primitive), so usage will look something like:
// 
// vertices[vertexOffset + indices[indexOffset + (gl_PrimitiveID * 3) + 0/1/2]] 
// materials[materialIndex]
struct GeometryInfo
{
    // Element offset into our vertex buffer.
    // This marks the start of the vertices that correspond to the
    // gltf primitive that this geometry row is using.
    uint vertexOffset;

    // Element offset into our index buffer.
    // This marks the start of the indices that correspond to the
    // gltf primitive that this geometry row is using.
    uint indexOffset;

    // Direct index into our material buffer.
    uint materialIndex;
};

struct GpuMaterial
{
    vec3  baseColor;
    uint  type;
    vec3  emissionColor;
    float roughness;
    float metallic;
    float ior;
    float transmission;
};

struct EmissiveTri
{
    vec3 v0;
    vec3 v1;
    vec3 v2;
    uint materialIndex;
    float area;
    float cdf;
};

// Device Buffer Addresses for our scene buffers
struct SceneAddresses
{
    // Starting address of our buffer of Vertex structs
    uint64_t verticesAddr;

    // Starting address of our buffer of uint32 Vertex indices
    uint64_t indicesAddr;

    // Starting address of our buffer of GeometryInfo structs
    uint64_t geometriesAddr;

    // Starting address of our buffer of GpuMaterial structs
    uint64_t materialsAddr;
    
    // Starting address of our buffer of EmissiveTri structs
    uint64_t lightsAddr;
    uint lightCount;
    float totalLightArea;
};

struct CameraParams
{
    vec3 position;
    vec3 view;
    vec3 right;
    vec3 up;
    vec2 pixelLength;
};

// Push constant limits are per-device, with 128 bytes guaranteed
// by Vulkan.
// 
// Running total: 4 * vec3 (48) + 3 * uint (12) + vec2 (8) + uint64 (8) = 76.
// Largest alignment: 8
// Total size with alignment padding: 80
//
// Additionally, because uint64 has alignment 8, it must start at a multiple of
// its alignment. The other members sum to 68, which is a multiple of 4 but not 8.
// If we place sceneAddr after the other members, 4 bytes of padding would be inserted
// after the other members and before sceneAddr. If we place sceneAddr at the beginning,
// that padding gets placed at the end of the struct, which makes all the data contiguous.
struct PushConstants
{
    // The device buffer address point to our single instance
    // of the SceneAddresses struct. We do this to keep the size
    // of PushConstants as small as possible so we don't risk hitting
    // the 128 byte soft cap
    uint64_t sceneAddr;
    CameraParams camera;
    uint renderedFrameCount;
    uint maxDepth;
    uint flags;
};

GPU_NAMESPACE_END

#ifdef __cplusplus
static_assert(sizeof(gpu::PushConstants) <= 128,
    "Vulkan only guarantees a max size of 128 bytes for push constants. We have exceeded that.");
#endif

#endif // GPU_SHARED_H
