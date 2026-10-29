#pragma once

#include "utilities.h"

#include <glm/glm.hpp>
#include <thrust/random.h>

// Seeded from (iter, index, depth), so a render repeats bit for bit.
__host__ __device__ inline thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1u << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Orthonormal frame around unit n (Duff et al. 2017, "Building an Orthonormal Basis, Revisited").
__host__ __device__ inline void frameAround(glm::vec3 n, glm::vec3& t, glm::vec3& b)
{
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = glm::vec3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = glm::vec3(c, sign + n.y * n.y * a, -n.y);
}

// Direction around unit n with density cos(theta) / pi.
__host__ __device__ inline glm::vec3 sampleCosineHemisphere(glm::vec3 n, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float cosTheta = sqrtf(u01(rng));
    const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    const float phi = TWO_PI * u01(rng);
    glm::vec3 t;
    glm::vec3 b;
    frameAround(n, t, b);
    return cosTheta * n + (sinTheta * cosf(phi)) * t + (sinTheta * sinf(phi)) * b;
}
