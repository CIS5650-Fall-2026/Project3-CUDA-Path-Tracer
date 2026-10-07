#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bxdf_utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float ggx_lambda(glm::vec3 w, float ax, float ay) {
    float x = w.x * ax;
    float y = w.y * ay;
    float q = sqrtf(1.f + (x * x + y * y) / (w.z * w.z));

    return (q - 1.f) / 2.f;
}

__device__ __forceinline__ glm::vec2 ggx_alpha(float anis, float rough) {
    float aspect = sqrtf(1.f - 0.9f * anis);
    const float amin = 0.0001f;
    float ax = glm::clamp(rough * rough / aspect, amin, 1.f);
    float ay = glm::clamp(rough * rough * aspect, amin, 1.f);
    return glm::vec2(ax, ay);
}

__device__ __forceinline__ float ggx_d(glm::vec3 wh, float ax, float ay) {
    float x = wh.x * wh.x / (ax * ax) + wh.y * wh.y / (ay * ay) + wh.z * wh.z;
    float denom = PI * ax * ay * x * x;
    return 1.f / denom;
}

struct GgxEval {
    float dg;
    float pdf;
    glm::vec3 wh;
};

__device__ __forceinline__ GgxEval eval_pdf_ggx(glm::vec3 wo, glm::vec3 wi, float ax, float ay) {
    GgxEval r{};
    if (bx::cos_theta(wo) * bx::cos_theta(wi) <= 0.f) {
        return r;
    }

    glm::vec3 h = wo + wi;
    float len2 = glm::dot(h, h);
    if (len2 <= 0.f) {
        return r;
    }
    h *= rsqrtf(len2);
    h = h.z < 0.f ? -h : h;

    float d = ggx_d(h, ax, ay);
    float lo = ggx_lambda(wo, ax, ay);
    float li = ggx_lambda(wi, ax, ay);
    float inv_4cos = 1.f / (4.f * bx::abs_cos(wo));

    r.dg = d / (1.f + lo + li) * inv_4cos;
    r.pdf = d / (1.f + lo) * inv_4cos;
    r.wh = h;
    return r;
}

// Dupuy & Benyoub 2023
__device__ __forceinline__ glm::vec3 sample_ggx_vndf(glm::vec3 wo, float ax, float ay,
                                                     RngEng& rng) {
    glm::vec3 wo_std = glm::normalize(glm::vec3(wo.x * ax, wo.y * ay, wo.z));

    glm::vec3 c = sample_uniform_sphere_cap(rng, wo_std.z);

    glm::vec3 wh_std = c + wo_std;
    return glm::normalize(glm::vec3(wh_std.x * ax, wh_std.y * ay, wh_std.z));
}
