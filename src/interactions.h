#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 calculateReflectedRayDirection(
    glm::vec3 wo,
    glm::vec3 normal);

__host__ __device__ glm::vec3 calculateRefractedRayDirection(
    glm::vec3 wo,
    glm::vec3 normal,
    float ri);

__host__ __device__ glm::vec3 metallicFresnel(
    glm::vec3 r0,
    float cosine);

__host__ __device__ float transmissiveFresnel(
    float r0,
    float cosine);

__host__ __device__ float smithGGXMaskingShadowing(
    glm::vec3 wi,
    glm::vec3 wo,
    float a2);

__host__ __device__ void sampleGGXNorm(
    const Material& m,
    glm::vec3 wo,
    glm::vec3& wi,
    glm::vec3& reflectance,
    thrust::default_random_engine& rng);

/*__host__ __device__ void uniformSampleOneLight(
    const int numLights,
    glm::vec3 point,
    thrust::default_random_engine& rng
);*/

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRayOpaque(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 texAlbedo,
    glm::vec3 texRough,
    Material& m,
    thrust::default_random_engine& rng);

__host__ __device__ void scatterRayTransparent(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    bool outside,
    thrust::default_random_engine& rng);

__host__ __device__ void scatterRayFake(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);