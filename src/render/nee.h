#pragma once

// Next event estimation on the device: light picking, light sampling, MIS weight.

#include <cuda_runtime.h>

#include "render/environment.h"
#include "render/pbr_surface.h"
#include "scene/sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

// The light list on the device. The pick tables have one row per receiver
// mask, ordered triangles, punctual lights, then the environment.
struct LightList
{
    const LightTriangle* triangles;
    const PunctualLight* punctual;
    const float* cdf;                 // numMasks * numLights
    const float* emitterAreaPdf;      // numMasks * numMaterials
    const float* punctualPickPdf;     // numMasks * numPunctual
    const float* environmentPickPdf;  // numMasks; 0 without an environment to sample
    const unsigned int* masks;        // the distinct masks, all ones first
    int numMasks;
    int numMaterials;
    int numPunctual;
    int numTriangles;
    int numLights;                    // triangles + punctual + environment; 0 when there is nothing to sample
};

// Pick table row for a mask; row 0 (all ones) stands in for a mask the tables do not hold.
__device__ __forceinline__ int lightRow(const LightList& lights, unsigned int mask)
{
    for (int m = 1; m < lights.numMasks; ++m)
    {
        if (lights.masks[m] == mask)
        {
            return m;
        }
    }
    return 0;
}

// Power heuristic, exponent 2 (Veach 1997), as a ratio so a huge density cannot overflow its square.
__device__ __forceinline__ float powerHeuristic(float pThis, float pOther)
{
    const float r = pOther / pThis;
    return 1.0f / (1.0f + r * r);
}

// Emission of a light triangle seen at cosLight (positive on its front face); alpha scales or cuts it.
__device__ __forceinline__ glm::vec3 lightEmission(const Material& m, glm::vec2 uv, float cosLight,
    const cudaTextureObject_t* textures)
{
    const float4 emissive = sampleSlot(textures, m.emissiveTexture, uv);
    glm::vec3 emission = m.emission * glm::vec3(emissive.x, emissive.y, emissive.z);
    if (hasClearcoat(m, cosLight > 0.0f))
    {
        emission *= 1.0f - m.clearcoat * schlickFresnel(glm::vec3(0.04f), fabsf(cosLight)).x;
    }
    if (m.alphaMode != ALPHA_OPAQUE)
    {
        float alpha = m.alpha;
        if (m.baseColorTexture >= 0)
        {
            alpha *= tex2D<float4>(textures[m.baseColorTexture], uv.x, uv.y).w;
        }
        if (m.alphaMode == ALPHA_MASK)
        {
            return alpha < m.alphaCutoff ? glm::vec3(0.0f) : emission;
        }
        emission *= alpha;
    }
    return emission;
}

__device__ __forceinline__ bool receivesGroup(unsigned int receiverMask, int group)
{
    return ((receiverMask >> group) & 1u) != 0u;
}

// Smoothstep of the cosine between the inner and outer cone, matching Cycles.
__device__ __forceinline__ float spotFalloff(const PunctualLight& light, glm::vec3 wi)
{
    const float cosAngle = -glm::dot(wi, light.direction);
    if (cosAngle <= light.cosOuter)
    {
        return 0.0f;
    }
    if (cosAngle >= light.cosInner)
    {
        return 1.0f;
    }
    const float t = (cosAngle - light.cosOuter) / (light.cosInner - light.cosOuter);
    return t * t * (3.0f - 2.0f * t);
}

struct LightSample
{
    glm::vec3 wi;             // unit direction toward the light
    float tMax;               // shadow ray length, short of the light itself
    glm::vec3 weightedLight;  // arriving light over the density of sampling it
    float pdf;                // solid-angle pdf of wi; 0 for a point or distant light
};

// Picks a light with u0 and a point or direction on it with u1, u2. False
// when the sample brings no light; a dropped sample is a zero, so the
// estimate stays unbiased.
__device__ __forceinline__ bool sampleLight(const LightList& lights, const EnvironmentMap& env,
    const Material* materials, const cudaTextureObject_t* textures, glm::vec3 hitPoint, unsigned int receiverMask,
    float u0, float u1, float u2, LightSample& sample)
{
    const int row = lightRow(lights, receiverMask);
    const int index = searchCdf(lights.cdf + row * lights.numLights, lights.numLights, u0);
    // Shadow rays stop short of the light by the float error at the larger point
    const auto margin = [&](glm::vec3 lightPoint, float distance) {
        const float scale = fmaxf(1.0f, fmaxf(maxComponent(glm::abs(hitPoint)), maxComponent(glm::abs(lightPoint))));
        return fmaxf(1e-4f * distance, 4.0f * EPSILON * scale);
    };

    if (index < lights.numTriangles)
    {
        const LightTriangle t = lights.triangles[index];
        if (!receivesGroup(receiverMask, t.group))
        {
            return false;
        }
        const float s = sqrtf(u1);
        const float b1 = 1.0f - s;
        const float b2 = u2 * s;
        const glm::vec3 p = t.p0 + b1 * t.e1 + b2 * t.e2;
        const glm::vec3 toLight = p - hitPoint;
        const float d2 = glm::dot(toLight, toLight);
        const float d = sqrtf(d2);
        const float tMax = d - margin(p, d);
        const glm::vec3 normal = glm::cross(t.e1, t.e2);
        const float cosLight = -glm::dot(normal, toLight) / (glm::length(normal) * d);
        if (!(tMax > 0.0f) || fabsf(cosLight) < 1e-6f)
        {
            return false;
        }
        const glm::vec2 uv = (1.0f - b1 - b2) * t.uv0 + b1 * t.uv1 + b2 * t.uv2;
        const glm::vec3 emission = lightEmission(materials[t.materialId], uv, cosLight, textures);
        if (emission == glm::vec3(0.0f))
        {
            return false;
        }
        sample.wi = toLight / d;
        sample.tMax = tMax;
        sample.pdf = lights.emitterAreaPdf[row * lights.numMaterials + t.materialId] * d2 / fabsf(cosLight);
        sample.weightedLight = emission / sample.pdf;
        return true;
    }

    const float environmentPick = lights.environmentPickPdf[row];
    if (environmentPick > 0.0f && index == lights.numLights - 1)
    {
        float pdf;
        if (!environmentSample(env, u1, u2, sample.wi, pdf))
        {
            return false;
        }
        sample.tMax = 1e16f;
        sample.pdf = environmentPick * pdf;
        sample.weightedLight = environmentRadiance(env, sample.wi) / sample.pdf;
        return true;
    }

    const PunctualLight light = lights.punctual[index - lights.numTriangles];
    if (!receivesGroup(receiverMask, light.group))
    {
        return false;
    }
    const float pickPdf = lights.punctualPickPdf[row * lights.numPunctual + index - lights.numTriangles];
    sample.pdf = 0.0f;
    if (light.type != LIGHT_DISTANT)
    {
        const glm::vec3 toLight = light.position - hitPoint;
        const float d2 = glm::dot(toLight, toLight);
        const float d = sqrtf(d2);
        sample.tMax = d - margin(light.position, d);
        if (!(sample.tMax > 0.0f))
        {
            return false;
        }
        sample.wi = toLight / d;
        sample.weightedLight = light.intensity / (d2 * pickPdf);
        if (light.type == LIGHT_SPOT)
        {
            const float falloff = spotFalloff(light, sample.wi);
            if (falloff <= 0.0f)
            {
                return false;
            }
            sample.weightedLight *= falloff;
        }
    }
    else
    {
        sample.wi = light.position;
        sample.tMax = 1e16f;  // the path rays' own tmax
        sample.weightedLight = light.intensity / pickPdf;
    }
    return true;
}
