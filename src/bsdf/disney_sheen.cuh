#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float luminance(glm::vec3 rgb) {
    return 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
}

__device__ __forceinline__ glm::vec3 sheen_c_tint(glm::vec3 col) {
    float lum = luminance(col);
    return lum > 0.f ? col / lum : glm::vec3(1);
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_sheen(glm::vec3 wo, glm::vec3 wi,
                                                          const DisneySheen& m) {
    BsdfEval eval{};
    if (bx::cos_theta(wo) * bx::cos_theta(wi) <= 0.f) {
        return eval;
    }

    glm::vec3 h = wo + wi;
    float len2 = glm::dot(h, h);
    if (len2 <= 0.f) {
        return eval;
    }
    h *= rsqrtf(len2);

    glm::vec3 ctint = sheen_c_tint(m.color);
    glm::vec3 c = (1.f - m.sheen_tint) + m.sheen_tint * ctint;

    float x = 1.f - bx::abs_dot(h, wo);
    x = x * x * x * x * x; // x^5
    eval.f = c * x * bx::abs_cos(wi);
    eval.pdf = bx::non_neg_cos(wi) * INV_PI;

    return eval;
}

__device__ __forceinline__ BsdfSample sample_disney_sheen(glm::vec3 wo, const DisneySheen& m,
                                                          RngEng& rng) {
    glm::vec3 wi = calculate_random_direction_in_cosine_hemisphere(rng);
    BsdfEval e = eval_pdf_disney_sheen(wo, wi, m);

    return BsdfSample{wi, e.pdf, e.f, BxdfFlag::Diffuse};
}
