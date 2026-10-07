#pragma once

#include "thrust_utils.h"
#include "math_utils.h"
#include <glm/glm.hpp>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__forceinline__ __host__ __device__ unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__forceinline__ __host__ __device__ RngEng make_seeded_rng(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return RngEng(h);
}

__forceinline__ __device__ glm::vec3 square_to_disk_concentric(glm::vec2 xi) {
    float x = xi.x * 2.f - 1.f;
    float y = xi.y * 2.f - 1.f;

    if (x == 0.f && y == 0.f) {
        return glm::vec3(0, 0, 0);
    }

    float r;
    float theta;

    if (fabsf(x) > fabsf(y)) {
        r = x;
        theta = PI / 4.f * y / x;
    } else {
        r = y;
        theta = PI / 2.f - (PI / 4.f * x / y);
    }

    return glm::vec3(r * cosf(theta), r * sinf(theta), 0);
}

__forceinline__ __device__ glm::vec3 square_to_hemisphere_cosine(glm::vec2 xi) {
    glm::vec3 disk = square_to_disk_concentric(xi);
    return glm::vec3(disk.x, disk.y, sqrtf(fmaxf(1 - disk.x * disk.x - disk.y * disk.y, 0.f)));
}

__forceinline__ __device__ float square_to_hemisphere_cosine_pdf(glm::vec3 s) {
    return s.z * INV_PI;
}

__forceinline__ __device__ glm::vec3 square_to_sphere_uniform(glm::vec2 xi) {
    float z = 1 - 2 * xi.x;
    float z_comp = sqrtf(1 - z * z);
    float x = cosf(2 * PI * xi.y) * z_comp;
    float y = sinf(2 * PI * xi.y) * z_comp;

    return glm::vec3(x, y, z);
}

__forceinline__ __device__ glm::vec3 square_to_sphere_cap(glm::vec2 xi, float wo_z) {
    float phi = 2.f * PI * xi.x;
    float z = (1.f - xi.y) * (1.f + wo_z) - wo_z;
    float sin_t = sqrtf(fmaxf(0.f, 1.f - z * z));
    return glm::vec3(sin_t * cosf(phi), sin_t * sinf(phi), z);
}

__forceinline__ __device__ float square_to_sphere_uniform_pdf(glm::vec3 s) {
    return INV_FOUR_PI;
}

__forceinline__ __device__ glm::vec3 calculate_random_direction_in_cosine_hemisphere(RngEng& rng) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));
    glm::vec3 dir = square_to_hemisphere_cosine(xi);

    return dir;
}

__forceinline__ __device__ glm::vec2 sample_uniform_disk(RngEng& rng) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));

    return glm::vec2(square_to_disk_concentric(xi));
}

// range [-wo_z, 1]
__forceinline__ __device__ glm::vec3 sample_uniform_sphere_cap(RngEng& rng, float wo_z) {
    UnifDist<float> u01(0, 1);

    glm::vec2 xi(u01(rng), u01(rng));

    return square_to_sphere_cap(xi, wo_z);
}
