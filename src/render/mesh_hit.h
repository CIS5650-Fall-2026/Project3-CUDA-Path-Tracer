#pragma once

// Mesh hit attributes shared by the naive mesh test and the OptiX hit
// programs, so both paths agree bit for bit. Header-only: the OptiX programs
// compile to OptiX-IR and cannot call into other .cu files.

#include <cuda_runtime.h>

#include "scene/sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

template <typename T>
__device__ __forceinline__ T interpolate(const T* attribute, glm::ivec3 tri, float u, float v)
{
    return (1.0f - u - v) * attribute[tri.x] + u * attribute[tri.y] + v * attribute[tri.z];
}

// World-space shading normal facing the ray; falls back to the geometric
// normal where a smooth vertex normal leans past the surface near a silhouette.
template <typename NormalToWorld>
__device__ __forceinline__ glm::vec3 meshShadingNormal(glm::vec3 vertexNormal, glm::vec3 geometricNormal, bool outside,
    glm::vec3 rayDirection, NormalToWorld normalToWorld)
{
    glm::vec3 normal = glm::normalize(normalToWorld(vertexNormal));
    if (!outside)
    {
        normal = -normal;
    }
    if (glm::dot(normal, rayDirection) > 0.0f)
    {
        normal = glm::normalize(normalToWorld(geometricNormal));
        if (!outside)
        {
            normal = -normal;
        }
    }
    return normal;
}

// World-space tangent; tangentSign flips w under a mirroring transform.
template <typename VectorToWorld>
__device__ __forceinline__ glm::vec4 meshTangent(const glm::vec4* tangents, glm::ivec3 tri, float u, float v,
    float tangentSign, VectorToWorld vectorToWorld)
{
    const glm::vec3 t(interpolate(tangents, tri, u, v));
    return glm::vec4(vectorToWorld(t), tangentSign * tangents[tri.x].w);
}

// Seed of a path segment's ALPHA_BLEND tests.
__device__ __forceinline__ unsigned int alphaPathSeed(int iter, int pixelIndex, int depth)
{
    return utilhash(utilhash(utilhash(pixelIndex) ^ iter) ^ depth);
}

// Seed of a shadow ray's ALPHA_BLEND tests; the flag bit keeps it apart from
// the path's own seed so the two decisions are not correlated.
__device__ __forceinline__ unsigned int shadowAlphaSeed(int iter, int pixelIndex, int depth)
{
    return alphaPathSeed(iter, pixelIndex, depth | 0x10000);
}

// Whether a hit falls in an alpha cut-out. ALPHA_BLEND passes the ray with
// probability 1 - alpha, decided by a hash so OptiX's repeated any-hit calls
// agree. The texture unit reads alpha linearly even from an sRGB texture.
__device__ __forceinline__ bool alphaCutOut(const Material& m, glm::vec2 uv, const cudaTextureObject_t* textures,
    unsigned int pathSeed, int instance, int triangle)
{
    if (m.alphaMode == ALPHA_OPAQUE)
    {
        return false;
    }
    float alpha = m.alpha;
    if (m.baseColorTexture >= 0)
    {
        alpha *= tex2D<float4>(textures[m.baseColorTexture], uv.x, uv.y).w;
    }
    if (m.alphaMode == ALPHA_MASK)
    {
        return alpha < m.alphaCutoff;
    }
    // The top 24 bits as a float in [0, 1): below 1, so alpha 1 never passes.
    const unsigned int h = utilhash(utilhash(pathSeed ^ instance) ^ triangle);
    return (h >> 8) * (1.0f / 16777216.0f) >= alpha;
}
