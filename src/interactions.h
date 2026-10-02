#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng,
    float hemisphereSample = -1.0f, float azimuthSample = -1.0f);

/**
 * Scatter a diffuse ray in a cosine-weighted hemisphere.
 * Offset its origin from the surface and multiply throughput by material color.
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);

// Explicit samples keep streams stable through sorting, compaction and resume.
__host__ __device__ bool scatterRay(PathSegment& path, glm::vec3 point,
    glm::vec3 normal, bool frontFace, const Material& material,
    bool refraction, float u, float v);
__host__ __device__ bool scatterSurface(PathSegment& path, glm::vec3 point,
    glm::vec3 normal, bool frontFace, const Material& material,
    bool refraction, float u, float v);
