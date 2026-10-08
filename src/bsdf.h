#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

__host__ __device__ glm::vec3 sampleDiffuse(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 sampleSpecularReflect(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    thrust::default_random_engine& rng);


__host__ __device__ float computeFresnelDielectric(
    float cosThetaI,
    float etaI,
    float etaT);

// Fresnel reflectance
// cosThetaI = dot(wo, N): >0 entering (etaI->etaT), <0 exiting 
// returns reflected fraction (1-result = transmitted) / 1 on TIR
__host__ __device__ glm::vec3 sampleDielectric(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    float ior,
    thrust::default_random_engine& rng);
