#pragma once

#include "sceneStructs.h"
#include <glm/glm.hpp>
#include <thrust/random.h>

// Cosine-weighted sampling about a unit world-space normal.
__host__ __device__ glm::vec3 sampleCosineWeightedHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng);

namespace bsdf {

// Local convention: normal = +Z, both directions point away from the hit,
// wo.z > 0. Reflection has wi.z > 0. transmission has wi.z < 0.
// frontFace records the original outward geometric normal's side; orienting
// the reflection frame cant erase which dielectric medium is incident.
struct Sample {
    glm::vec3 wi = glm::vec3(0.0f);
    glm::vec3 f = glm::vec3(0.0f); // BRDF.
    float pdf = 0.0f;             // mixture density
    bool delta = false;
    bool transmission = false;
    float branchProbability = 0.0f; // Discrete mass, not a solid-angle PDF.
    glm::vec3 deltaWeight = glm::vec3(0.0f); // already divided by branch mass.
};

// PBRT v4 sections 9.3 and 9.5:  Fresnel, Snell, radiance transport.
// Exposing branch variate makes entry,exit,TIR checks deterministic.
__host__ __device__ Sample sampleSmoothDielectric(
    const Material& material, const glm::vec3& wo, bool frontFace, float u);

__host__ __device__ glm::vec3 evaluate(
    const Material& material, const glm::vec3& wo, const glm::vec3& wi,
    bool frontFace);

__host__ __device__ float pdf(
    const Material& material, const glm::vec3& wo, const glm::vec3& wi);

// Continuous pdf == 0, or delta branchProbability == 0, is invalid. no retry.
__host__ __device__ Sample sample(
    const Material& material, const glm::vec3& wo, bool frontFace,
    thrust::default_random_engine& rng);

} // namespace bsdf

// 'normal' is the outward geometric normal. Builds an oriented frame,
// uses deltaWeight or f * abs(cos) / pdf and creates one offset continuation.
// An absorbing, invalid or exhausted path is terminated with black color.
// Emission is handled b4 calling this function, in shadeMaterial.
__host__ __device__ void scatterRay(
    PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal,
    const Material& material, thrust::default_random_engine& rng);
