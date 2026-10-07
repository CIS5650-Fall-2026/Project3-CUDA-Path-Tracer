#pragma once

#include "bsdf_structs.cuh"
#include "conductor.cuh"
#include "disney_sheen.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Gulbrandsen 2014
__host__ __device__ __forceinline__ Conductor to_conductor(const DisneyMetal& m) {
    glm::vec3 r = glm::clamp(m.color, glm::vec3(0.f), glm::vec3(0.99f));
    glm::vec3 g = glm::clamp(m.edge_tint, glm::vec3(0.f), glm::vec3(1.f));
    glm::vec3 sr = glm::sqrt(r);

    glm::vec3 n_min = (1.f - r) / (1.f + r);
    glm::vec3 n_max = (1.f + sr) / (1.f - sr);
    glm::vec3 n = g * n_min + (1.f - g) * n_max;

    glm::vec3 np1 = n + 1.f;
    glm::vec3 nm1 = n - 1.f;
    glm::vec3 k2 = (r * np1 * np1 - nm1 * nm1) / (1.f - r);
    glm::vec3 k = glm::sqrt(glm::max(k2, glm::vec3(0.f)));

    return Conductor{n, k, m.roughness, m.anisotropic};
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_metal(glm::vec3 wo, glm::vec3 wi,
                                                          const DisneyMetal& m) {
    return eval_pdf_conductor(wo, wi, to_conductor(m));
}

__device__ __forceinline__ BsdfSample sample_disney_metal(glm::vec3 wo, const DisneyMetal& m,
                                                          RngEng& rng) {
    return sample_conductor(wo, to_conductor(m), rng);
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_metal_uber(glm::vec3 wo, glm::vec3 wi,
                                                               const DisneyBsdf& m) {

    if (bx::cos_theta(wo) <= 0.f || bx::cos_theta(wi) <= 0.f) {
        return BsdfEval{};
    }

    float r = fmaxf(m.roughness, 0.01f);
    glm::vec2 a = ggx_alpha(m.anisotropic, r);
    GgxEval g = eval_pdf_ggx(wo, wi, a.x, a.y);
    if (g.pdf == 0.f) {
        return BsdfEval{};
    }

    float rn = m.ior - 1.f;
    float rd = m.ior + 1.f;
    float r0 = (rn * rn) / (rd * rd);

    float c = bx::abs_dot(wo, g.wh);
    float x = 1.f - c;
    x = x * x * x * x * x; // x^5

    glm::vec3 ks = (1.f - m.specular_tint) + m.specular_tint * sheen_c_tint(m.color);
    glm::vec3 c0 = m.specular * r0 * ks * (1.f - m.metallic) + m.metallic * m.color;
    glm::vec3 f_diel = c0 + (1.f - c0) * x;

    Conductor cd = to_conductor(DisneyMetal{m.color, m.edge_tint, r, m.anisotropic});
    glm::vec3 f_cond = fresnel_conductor_eval(c, glm::vec3(1.f), cd.eta, cd.k);

    glm::vec3 fr = (1.f - m.metallic) * f_diel + m.metallic * f_cond;
    return BsdfEval{fr * g.dg, g.pdf};
}

__device__ __forceinline__ BsdfSample sample_disney_metal_uber(glm::vec3 wo, const DisneyBsdf& m,
                                                               RngEng& rng) {
    float r = fmaxf(m.roughness, 0.01f);
    glm::vec2 a = ggx_alpha(m.anisotropic, r);
    glm::vec3 wh = sample_ggx_vndf(wo, a.x, a.y, rng);
    glm::vec3 wi = glm::reflect(-wo, wh);
    if (bx::cos_theta(wi) <= 0.f) {
        return BsdfSample{};
    }

    BsdfEval e = eval_pdf_disney_metal_uber(wo, wi, m);
    return BsdfSample{wi, e.pdf, e.f, BxdfFlag::Glossy | BxdfFlag::Reflection};
}
