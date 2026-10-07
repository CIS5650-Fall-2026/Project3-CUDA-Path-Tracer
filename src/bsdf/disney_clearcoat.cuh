#pragma once

#include "../math_utils.h"
#include "../thrust_utils.h"
#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float c_lambda(glm::vec3 w) {
    constexpr float m = 0.25f;
    float x = w.x * m;
    float y = w.y * m;
    float q = sqrtf(1.f + (x * x + y * y) / (w.z * w.z));

    return (q - 1.f) * 0.5f;
}

__device__ __forceinline__ float cc_alpha(float gloss) {
    return (1.f - gloss) * 0.1f + gloss * 0.001f;
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_clearcoat(glm::vec3 wo, glm::vec3 wi,
                                                              const DisneyClearcoat& m) {
    BsdfEval eval{};
    if (bx::cos_theta(wo) * bx::cos_theta(wi) <= 0.f) {
        return eval;
    }

    constexpr float ior = 1.5f;

    float a = cc_alpha(m.gloss);

    glm::vec3 h = wo + wi;
    float len2 = glm::dot(h, h);
    if (len2 <= 0.f) {
        return eval;
    }
    h *= rsqrtf(len2);

    float gi = 1.f / (1.f + c_lambda(wi));
    float go = 1.f / (1.f + c_lambda(wo));
    float g = gi * go;

    float x = 1.f + (a * a - 1.f) * h.z * h.z;
    float denom = PI * __logf(a * a) * x;
    float d = (a * a - 1.f) / denom;

    float rn = ior - 1.f;
    float rd = ior + 1.f;
    float r0 = (rn * rn) / (rd * rd);
    float p = (1.f - bx::abs_dot(h, wo));
    p = p * p * p * p * p; // p^5

    float f = r0 + (1.f - r0) * p;

    eval.f = glm::vec3(f * d * g) / (4.f * bx::abs_cos(wo));
    eval.pdf = eval.pdf = d * bx::abs_cos(h) / (4.f * bx::abs_dot(wo, h));

    return eval;
}

__device__ __forceinline__ BsdfSample sample_disney_clearcoat(glm::vec3 wo,
                                                              const DisneyClearcoat& m,
                                                              RngEng& rng) {
    UnifDist<float> u01(0, 1);
    glm::vec2 xi(u01(rng), u01(rng));

    float a = cc_alpha(m.gloss);

    float cos_e2 = (1.f - __powf(a * a, 1.f - xi.x)) / (1.f - a * a);
    float ce = sqrtf(cos_e2);
    float se = sqrtf(1.f - cos_e2);
    float sa, ca;
    sincospif(2.f * xi.y, &sa, &ca);

    glm::vec3 wh;
    wh.x = se * ca;
    wh.y = se * sa;
    wh.z = ce;

    glm::vec3 wi = glm::reflect(-wo, wh);
    if (bx::cos_theta(wi) <= 0.f) {
        return BsdfSample{};
    }

    BsdfEval e = eval_pdf_disney_clearcoat(wo, wi, m);
    return BsdfSample{wi, e.pdf, e.f, BxdfFlag::Glossy | BxdfFlag::Reflection};
}
