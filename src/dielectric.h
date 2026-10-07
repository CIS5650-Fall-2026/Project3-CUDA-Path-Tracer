#pragma once

#include "interactions.h"

#include <thrust/random.h>

namespace Dielectric
{
__device__ float fresnel_dielectric(float cosThetaI, float etaI, float etaT);

__device__ ScatterResult calculate_transmission(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& material,
    bool smooth,
    glm::vec3 woWorld,
    const glm::vec3& sampledWm,
    thrust::default_random_engine& rng);

__device__ glm::vec3 get_volume_transmittance(
    const glm::vec3& attenuationColor,
    float attenuationDistance,
    float distance);
}
