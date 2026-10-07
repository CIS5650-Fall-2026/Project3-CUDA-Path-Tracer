#pragma once

#include "fresnel.cuh"

#include "../thrust_utils.h"
#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include "microfacet.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// assumes the medium is air and rgb approx, not spectral
__device__ __forceinline__ BsdfSample sample_smooth_conductor(glm::vec3 wo, const Conductor& m) {
    BsdfSample sample;

    sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
    sample.pdf = 1.f;
    // eta_i for air is 1
    glm::vec3 fr = fresnel_conductor_eval(bx::cos_theta(sample.wi), glm::vec3(1), m.eta, m.k);
    sample.f = glm::vec3(fr);
    sample.type = BxdfFlag::Reflection | BxdfFlag::Specular;

    return sample;
}

__device__ __forceinline__ BsdfEval eval_pdf_conductor(glm::vec3 wo, glm::vec3 wi,
                                                      const Conductor& m) {
    if (m.roughness == 0.f) {
        return BsdfEval{};
    }

    glm::vec2 a = ggx_alpha(m.anisotropy, m.roughness);
    GgxEval g = eval_pdf_ggx(wo, wi, a.x, a.y);
    if (g.pdf == 0.f) {
        return BsdfEval{};
    }

    glm::vec3 fr = fresnel_conductor_eval(glm::dot(wo, g.wh), glm::vec3(1.f), m.eta, m.k);
    return BsdfEval{fr * g.dg, g.pdf};
}

__device__ __forceinline__ BsdfSample sample_conductor(glm::vec3 wo, const Conductor& m,
                                                       RngEng& rng) {
    if (m.roughness == 0.f) {
        return sample_smooth_conductor(wo, m);
    }

    glm::vec2 a = ggx_alpha(m.anisotropy, m.roughness);
    glm::vec3 wh = sample_ggx_vndf(wo, a.x, a.y, rng);
    glm::vec3 wi = glm::reflect(-wo, wh);
    if (bx::cos_theta(wi) <= 0.f) {
        return BsdfSample{};
    }

    BsdfEval e = eval_pdf_conductor(wo, wi, m);
    return BsdfSample{wi, e.pdf, e.f, BxdfFlag::Glossy | BxdfFlag::Reflection};
}
