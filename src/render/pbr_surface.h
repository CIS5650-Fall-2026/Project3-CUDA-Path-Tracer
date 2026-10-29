#pragma once

// A material's inputs at one hit: texture lookups, normal map, emission.

#include <cuda_runtime.h>

#include "scene/sceneStructs.h"

#include <glm/glm.hpp>

struct PbrSurface
{
    glm::vec3 baseColor;
    float metallic;
    float roughness;
    float transmission;
    glm::vec3 emission;
    glm::vec3 normal;      // normal-mapped, facing the incoming ray
    glm::vec3 coatNormal;  // the mesh normal; the normal map does not touch the clearcoat
};

// A texture slot's value at uv, or 1 when the slot is empty.
__device__ __forceinline__ float4 sampleSlot(const cudaTextureObject_t* textures, int slot, glm::vec2 uv)
{
    return slot >= 0 ? tex2D<float4>(textures[slot], uv.x, uv.y) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}

// Normal-mapped world normal facing the ray. The frame is built on the
// front-face normal and flipped for a back-face hit; falls back to the
// unmapped normal for a degenerate tangent or a mapped normal facing away.
__device__ __forceinline__ glm::vec3 mappedNormal(const ShadeableIntersection& isect, float4 texel, float scale,
    glm::vec3 wo)
{
    const glm::vec3 n = isect.outside ? isect.surfaceNormal : -isect.surfaceNormal;
    const glm::vec3 tangent(isect.tangent);
    glm::vec3 t = tangent - n * glm::dot(n, tangent);
    const float length2 = glm::dot(t, t);
    if (length2 < 1e-12f)
    {
        return isect.surfaceNormal;
    }
    t *= rsqrtf(length2);
    const glm::vec3 b = isect.tangent.w * glm::cross(n, t);

    const float x = (2.0f * texel.x - 1.0f) * scale;
    const float y = (2.0f * texel.y - 1.0f) * scale;
    const float z = 2.0f * texel.z - 1.0f;
    glm::vec3 mapped = glm::normalize(x * t + y * b + z * n);
    if (!isect.outside)
    {
        mapped = -mapped;
    }
    return glm::dot(mapped, wo) > 0.0f ? mapped : isect.surfaceNormal;
}

// The coat lies on the air side: no coat inside a transmissive material.
__host__ __device__ __forceinline__ bool hasClearcoat(const Material& m, bool outside)
{
    return m.clearcoat > 0.0f && (outside || m.transmission == 0.0f);
}

// Schlick's Fresnel approximation with f90 = 1.
__host__ __device__ __forceinline__ glm::vec3 schlickFresnel(glm::vec3 f0, float cosTheta)
{
    const float m = glm::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    const float m2 = m * m;
    return f0 + (glm::vec3(1.0f) - f0) * (m2 * m2 * m);
}

// Emission at a hit, darkened by the clearcoat above it (KHR_materials_clearcoat).
__device__ __forceinline__ glm::vec3 pbrEmission(const Material& m, const ShadeableIntersection& isect, glm::vec3 wo,
    const cudaTextureObject_t* textures)
{
    const float4 emissive = sampleSlot(textures, m.emissiveTexture, isect.uv);
    glm::vec3 emission = m.emission * glm::vec3(emissive.x, emissive.y, emissive.z);
    if (hasClearcoat(m, isect.outside))
    {
        const float coatFresnel = schlickFresnel(glm::vec3(0.04f), glm::dot(isect.surfaceNormal, wo)).x;
        emission *= 1.0f - m.clearcoat * coatFresnel;
    }
    return emission;
}

// wo points from the hit back along the incoming ray.
__device__ __forceinline__ PbrSurface pbrSurface(const Material& m, const ShadeableIntersection& isect, glm::vec3 wo,
    const cudaTextureObject_t* textures)
{
    const glm::vec2 uv = isect.uv;
    PbrSurface s;

    const float4 base = sampleSlot(textures, m.baseColorTexture, uv);
    s.baseColor = m.baseColor * glm::vec3(base.x, base.y, base.z);

    // glTF packs roughness in G and metallic in B
    const float4 mr = sampleSlot(textures, m.metallicRoughnessTexture, uv);
    s.roughness = glm::clamp(m.roughness * mr.y, 0.0f, 1.0f);
    s.metallic = glm::clamp(m.metallic * mr.z, 0.0f, 1.0f);
    s.transmission = glm::clamp(m.transmission * sampleSlot(textures, m.transmissionTexture, uv).x, 0.0f, 1.0f);

    s.emission = pbrEmission(m, isect, wo, textures);

    s.coatNormal = isect.surfaceNormal;
    s.normal = isect.surfaceNormal;
    if (m.normalTexture >= 0 && isect.tangent.w != 0.0f)
    {
        s.normal = mappedNormal(isect, tex2D<float4>(textures[m.normalTexture], uv.x, uv.y), m.normalScale, wo);
    }
    return s;
}
