#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float dd_fresnel(glm::vec3 wo, glm::vec3 w, glm::vec3 wh, float a) {
    float c = bx::abs_dot(wh, wo);
    float d90 = 0.5f + 2.f * a * c * c;
    float x = (1.f - bx::abs_cos(w));
    x = x * x * x * x * x; // x^5
    return 1.f + (d90 - 1.f) * x;
}

__device__ __forceinline__ float dd_subsurface(glm::vec3 wo, glm::vec3 w, glm::vec3 wh, float a) {
    float c = bx::abs_dot(wh, wo);
    float fss90 = a * c * c;
    float x = (1.f - bx::abs_cos(w));
    x = x * x * x * x * x; // x^5
    return 1.f + (fss90 - 1.f) * x;
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_diffuse(glm::vec3 wo, glm::vec3 wi,
                                                            const DisneyDiffuse& m) {
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

    float fi = dd_fresnel(wo, wi, h, m.roughness);
    float fo = dd_fresnel(wo, wo, h, m.roughness);

    float c = INV_PI * bx::non_neg_cos(wi);
    glm::vec3 diffuse = m.color * fi * fo * c;

    float fssi = dd_subsurface(wo, wi, h, m.roughness);
    float fsso = dd_subsurface(wo, wo, h, m.roughness);

    float y = bx::abs_cos(wi) + bx::abs_cos(wo);
    float x = fssi * fsso * (1.f / y - 0.5f) + 0.5f;

    glm::vec3 subsurface = m.color * 1.25f * INV_PI * x * bx::abs_cos(wi);

    eval.f = (1.f - m.subsurface) * diffuse + m.subsurface * subsurface;
    eval.pdf = c;

    return eval;
}
__device__ __forceinline__ BsdfSample sample_disney_diffuse(glm::vec3 wo, const DisneyDiffuse& m,
                                                            RngEng& rng) {
    glm::vec3 wi = calculate_random_direction_in_cosine_hemisphere(rng);
    BsdfEval e = eval_pdf_disney_diffuse(wo, wi, m);
    return {wi, e.pdf, e.f, BxdfFlag::Diffuse};
}
