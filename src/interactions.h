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

// returns bsdf
// populate pdf and pathSegment (wi) 
__host__ __device__ glm::vec3 scatterRay(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    float ior, 
    thrust::default_random_engine& rng);
