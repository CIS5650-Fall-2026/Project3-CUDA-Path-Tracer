#pragma once

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include "dielectric.cuh"
#include "microfacet.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ BsdfEval eval_pdf_disney_glass(glm::vec3 wo, glm::vec3 wi,
                                                          const DisneyGlass& m) {
    BsdfEval eval{};

    float cos_o = bx::cos_theta(wo);
    float cos_i = bx::cos_theta(wi);
    if (cos_o == 0.f || cos_i == 0.f) {
        return eval;
    }

    bool reflect = cos_o * cos_i > 0.f;
    float etap = 1.f;

    if (!reflect) {
        // snells law for refraction
        etap = cos_o > 0.f ? m.ior : 1.f / m.ior;
    }

    glm::vec3 wm = wi * etap + wo;
    float len2 = glm::dot(wm, wm);
    if (len2 <= 0.f) {
        return eval;
    }
    wm *= rsqrtf(len2);
    wm = bx::cos_theta(wm) < 0.f ? -wm : wm;

    if (glm::dot(wm, wi) * bx::cos_theta(wi) < 0.f || glm::dot(wm, wo) * bx::cos_theta(wo) < 0.f) {
        // back facing microfacets
        return eval;
    }

    glm::vec2 a = ggx_alpha(m.anisotropic, m.roughness);
    float d = ggx_d(wm, a.x, a.y);
    float lo = ggx_lambda(wo, a.x, a.y);
    float li = ggx_lambda(wi, a.x, a.y);
    float g = 1.f / (1.f + lo + li);
    float g1 = 1.f / (1.f + lo);
    float f = fresnel_dielectric_eval(glm::dot(wo, wm), 1.f, m.ior);

    float abs_cos_o = fabsf(cos_o);
    float wo_wm = glm::dot(wo, wm);
    float wi_wm = glm::dot(wi, wm);

    float inv_cos_o = 1.f / abs_cos_o;

    if (reflect) {
        eval.f = f * d * g * inv_cos_o * 0.25f * m.color;
        eval.pdf = g1 * d * inv_cos_o * .25f * f;
    } else {
        float denom = wi_wm + wo_wm / etap;
        if (denom == 0.f)
            return eval;
        denom *= denom;
        float x = fabsf(wi_wm) / denom;
        eval.f = (1.f - f) * d * g * fabsf(wo_wm) * x * inv_cos_o * glm::sqrt(m.color);
        eval.f /= etap * etap;
        eval.pdf = g1 * d * fabsf(wo_wm) * inv_cos_o * x * (1.f - f);
    }

    return eval;
}

__device__ __forceinline__ BsdfSample sample_disney_glass(glm::vec3 wo, const DisneyGlass& m,
                                                          RngEng& rng) {

    if (m.roughness == 0.f) {
        BsdfSample sample = sample_smooth_dielectric(wo, m.ior, rng);
        if ((sample.type & BxdfFlag::Transmission) != BxdfFlag::Unset) {
            sample.f *= glm::sqrt(m.color);
        } else if ((sample.type & BxdfFlag::Reflection) != BxdfFlag::Unset) {
            sample.f *= m.color;
        }
        return sample;
    }

    glm::vec2 a = ggx_alpha(m.anisotropic, m.roughness);
    glm::vec3 wm = sample_ggx_vndf(bx::cos_theta(wo) < 0.f ? -wo : wo, a.x, a.y, rng);
    float f = fresnel_dielectric_eval(glm::dot(wo, wm), 1.f, m.ior);

    UnifDist<float> u01(0, 1);
    glm::vec3 wi;
    BxdfFlag type;
    if (u01(rng) < f) {
        // reflect
        wi = glm::reflect(-wo, wm);
        if (bx::cos_theta(wo) * bx::cos_theta(wi) <= 0.f) {
            return BsdfSample{};
        }
        type = BxdfFlag::Glossy | BxdfFlag::Reflection;
    } else {
        // refract
        float etap = bx::cos_theta(wo) > 0.f ? m.ior : 1.f / m.ior;
        if (!bx::refract(wo, bx::face_forward(wm, wo), 1.f / etap, wi)) {
            // total internal reflection
            return BsdfSample{};
        }
        if (bx::cos_theta(wo) * bx::cos_theta(wi) >= 0.f) {
            return BsdfSample{};
        }
        type = BxdfFlag::Glossy | BxdfFlag::Transmission;
    }

    BsdfEval e = eval_pdf_disney_glass(wo, wi, m);
    if (e.pdf == 0.f) {
        return BsdfSample{};
    }
    return BsdfSample{wi, e.pdf, e.f, type};
}
