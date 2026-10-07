#pragma once

#include "disney_clearcoat.cuh"
#include "disney_diffuse.cuh"
#include "disney_glass.cuh"
#include "disney_metal.cuh"
#include "disney_sheen.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct DisneyLobes {
    float w_diffuse;
    float w_sheen;
    float w_metal;
    float w_clearcoat;
    float w_glass;

    float p_diffuse;
    float p_metal;
    float p_clearcoat;
    float p_glass;
};

__device__ __forceinline__ DisneyLobes disney_lobes(const DisneyBsdf& m, bool inside) {
    DisneyLobes l{};
    if (inside) {
        l.w_glass = (1.f - m.metallic) * m.specular_transmission;
        l.p_glass = 1.f;
        return l;
    }

    l.w_diffuse = (1.f - m.specular_transmission) * (1.f - m.metallic);
    l.w_sheen = (1.f - m.metallic) * m.sheen;
    l.w_metal = 1.f - m.specular_transmission * (1.f - m.metallic);
    l.w_clearcoat = 0.25f * m.clearcoat;
    l.w_glass = (1.f - m.metallic) * m.specular_transmission;

    float sum = l.w_diffuse + l.w_metal + l.w_clearcoat + l.w_glass;
    if (sum <= 0.f) {
        return l;
    }
    float inv = 1.f / sum;
    l.p_diffuse = l.w_diffuse * inv;
    l.p_metal = l.w_metal * inv;
    l.p_clearcoat = l.w_clearcoat * inv;
    l.p_glass = l.w_glass * inv;
    return l;
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_bsdf(glm::vec3 wo, glm::vec3 wi,
                                                         const DisneyBsdf& m) {
    BsdfEval eval{};

    bool inside = bx::cos_theta(wo) < 0.f;
    float r = fmaxf(m.roughness, 0.01f);
    DisneyLobes lobes = disney_lobes(m, inside);

    if (lobes.w_diffuse > 0.f) {
        BsdfEval diffuse_eval =
            eval_pdf_disney_diffuse(wo, wi, DisneyDiffuse{m.color, r, m.subsurface});
        eval.f += lobes.w_diffuse * diffuse_eval.f;
        eval.pdf += lobes.p_diffuse * diffuse_eval.pdf;
    }
    if (lobes.w_sheen > 0.f) {
        BsdfEval sheen_eval = eval_pdf_disney_sheen(wo, wi, DisneySheen{m.color, m.sheen_tint});
        eval.f += lobes.w_sheen * sheen_eval.f;
    }
    if (lobes.w_metal > 0.f) {
        BsdfEval metal_eval =
            eval_pdf_disney_metal_uber(wo, wi, m);
        eval.f += lobes.w_metal * metal_eval.f;
        eval.pdf += lobes.p_metal * metal_eval.pdf;
    }
    if (lobes.w_clearcoat > 0.f) {
        BsdfEval clearcoat_eval =
            eval_pdf_disney_clearcoat(wo, wi, DisneyClearcoat{m.clearcoat_gloss});
        eval.f += lobes.w_clearcoat * clearcoat_eval.f;
        eval.pdf += lobes.p_clearcoat * clearcoat_eval.pdf;
    }
    if (lobes.w_glass > 0.f) {
        BsdfEval glass_eval =
            eval_pdf_disney_glass(wo, wi, DisneyGlass{m.color, r, m.anisotropic, m.ior});
        eval.f += lobes.w_glass * glass_eval.f;
        eval.pdf += lobes.p_glass * glass_eval.pdf;
    }

    return eval;
}

__device__ __forceinline__ BsdfSample sample_disney_bsdf(glm::vec3 wo, const DisneyBsdf& m,
                                                         RngEng& rng) {
    UnifDist<float> u01(0, 1);

    float u = u01(rng);

    bool inside = bx::cos_theta(wo) < 0.f;
    float r = fmaxf(m.roughness, 0.01f);
    DisneyLobes lobes = disney_lobes(m, inside);

    // skip sheen in the weights and sampling because it is covered by diffuse
    BsdfSample s;
    if (u < lobes.p_diffuse) {
        s = sample_disney_diffuse(wo, DisneyDiffuse{m.color, r, m.subsurface}, rng);
    } else if (u < lobes.p_diffuse + lobes.p_metal) {
        s = sample_disney_metal_uber(wo, m, rng);
    } else if (u < lobes.p_diffuse + lobes.p_metal + lobes.p_clearcoat) {
        s = sample_disney_clearcoat(wo, DisneyClearcoat{m.clearcoat_gloss}, rng);
    } else {
        s = sample_disney_glass(wo, DisneyGlass{m.color, r, m.anisotropic, m.ior}, rng);
    }
    if (s.pdf == 0.f) {
        return BsdfSample{};
    }
    BsdfEval e = eval_pdf_disney_bsdf(wo, s.wi, m);
    s.pdf = e.pdf;
    s.f = e.f;
    return s;
}
