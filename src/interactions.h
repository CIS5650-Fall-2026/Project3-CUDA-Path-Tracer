#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

__host__ __device__ inline unsigned int sampleHash(unsigned int x)
{
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

__host__ __device__ inline float hashToUnitFloat(unsigned int x)
{
    return (sampleHash(x) & 0x00FFFFFFu) / 16777216.0f;
}

__host__ __device__ inline float radicalInverse(unsigned int index, unsigned int base)
{
    float result = 0.0f;
    float inverseBase = 1.0f / (float)base;
    float fraction = inverseBase;

    while (index > 0)
    {
        unsigned int digit = index % base;
        result += (float)digit * fraction;
        index /= base;
        fraction *= inverseBase;
    }

    return result;
}

__host__ __device__ inline glm::vec2 halton2D(unsigned int index, unsigned int baseX, unsigned int baseY)
{
    float x = radicalInverse(index, baseX);
    float y = radicalInverse(index, baseY);
    return glm::vec2(x, y);
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);


__host__ __device__ glm::vec3 calculateSampledDirectionInHemisphere(glm::vec3 normal, float sample1, float sample2);
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
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    bool outside,
    bool enableRefraction,
    bool useLowDiscrepancy, 
    int iter, 
    int depth,
    thrust::default_random_engine& rng);


__device__ bool rayAABBIntersection(
    const Ray& ray, 
    const glm::vec3& minBounds, 
    const glm::vec3& maxBounds, 
    float maxT);