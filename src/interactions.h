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
/**
 * Sample diffuse bsdf
 */
__host__ __device__ glm::vec3 sampleDiffuse(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng);

/**
 * Two following ways to compute the fresnel term
 */
__host__ __device__ float computeFresnelReflectance(float cosThetaI, float eta); // more accurate physically

__host__ __device__ float computeSchlickApproxF(float cosThetaI, float eta); // fast approximation

/**
 * Samples a dielectric material, returns bsdf from it
 */
__host__ __device__ glm::vec3 sampleDielectric(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng);

/**
 * Samples a reflective material, returns bsdf from it
 */
__host__ __device__ glm::vec3 sampleReflection(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);

/**
 * Calculates direct lighting at a particular given ray
 */
__host__ __device__ glm::vec3 sampleDirectLighting(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    Geom* geoms,
    int geoms_size,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    Light* l,
    int lights_size,
    thrust::default_random_engine& rng
);
/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.

 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    Geom* geoms,
    int geoms_size,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    Light* l,
    int lights_size,
    thrust::default_random_engine& rng);
