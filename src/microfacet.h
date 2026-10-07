#pragma once

#include "interactions.h"

namespace Microfacet
{
__device__ float material_alpha(const Material& material);

__device__ void material_lobe_weights(
    const Material& material,
    float& wDiffuse,
    float& wMetal,
    float& wTransmission);

__host__ __device__ glm::vec3 sample_wm(float alpha, const glm::vec2& u);

__host__ __device__ glm::vec3 sample_wm(
    float alpha,
    const glm::vec3& normal,
    const glm::vec2& u);

__host__ __device__ ScatterResult get_brdf_result(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& material,
    float alpha,
    const glm::vec3& sampledWm,
    thrust::default_random_engine& rng);

__device__ ScatterResult calculate_reflection(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& material,
    bool pureDeltaMetal,
    const glm::vec3& sampledWm,
    thrust::default_random_engine& rng);

__host__ __device__ ScatterResult get_brdf_result(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& material,
    float alpha,
    thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 evaluate_brdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& material,
    float alpha);

__host__ __device__ float evaluate_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    float alpha);

__host__ __device__ float wm_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wmWorld,
    const glm::vec3& normal,
    float alpha);

__host__ __device__ glm::vec3 evaluate_dielectric_bsdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& material,
    bool outside,
    float alpha);

__host__ __device__ float evaluate_dielectric_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& material,
    bool outside,
    float alpha);
}
