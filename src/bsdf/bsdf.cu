#include "bsdf.cuh"
#include "bxdf_utils.cuh"
#include "conductor.cuh"
#include "dielectric.cuh"
#include "diffuse.cuh"
#include "disney_diffuse.cuh"
#include "disney_bsdf.cuh"
#include "disney_clearcoat.cuh"
#include "disney_glass.cuh"
#include "disney_metal.cuh"
#include "disney_sheen.cuh"

struct EvalPdfOp {
    glm::vec3 wo, wi;
    __device__ BsdfEval operator()(const Lambertian& m) const {
        return eval_pdf_diffuse(wi, m);
    }
    __device__ BsdfEval operator()(const Conductor& m) const {
        return eval_pdf_conductor(wo, wi, m);
    }
    __device__ BsdfEval operator()(const Dielectric& m) const {
        return eval_pdf_dielectric(wo, wi, m);
    }
    __device__ BsdfEval operator()(const DisneyDiffuse& m) const {
        return eval_pdf_disney_diffuse(wo, wi, m);
    }
    __device__ BsdfEval operator()(const DisneyMetal& m) const {
        return eval_pdf_disney_metal(wo, wi, m);
    }
    __device__ BsdfEval operator()(const DisneyClearcoat& m) const {
        return eval_pdf_disney_clearcoat(wo, wi, m);
    }
    __device__ BsdfEval operator()(const DisneyGlass& m) const {
        return eval_pdf_disney_glass(wo, wi, m);
    }
    __device__ BsdfEval operator()(const DisneySheen& m) const {
        return eval_pdf_disney_sheen(wo, wi, m);
    }
    __device__ BsdfEval operator()(const DisneyBsdf& m) const {
        return eval_pdf_disney_bsdf(wo, wi, m);
    }
};

struct SampleOp {
    glm::vec3 wo;
    RngEng& rng;
    __device__ BsdfSample operator()(const Lambertian& m) const {
        return sample_diffuse(m, rng);
    }
    __device__ BsdfSample operator()(const Conductor& m) const {
        return sample_conductor(wo, m, rng);
    }
    __device__ BsdfSample operator()(const Dielectric& m) const {
        return sample_dielectric(wo, m, rng);
    }
    __device__ BsdfSample operator()(const DisneyDiffuse& m) const {
        return sample_disney_diffuse(wo, m, rng);
    }
    __device__ BsdfSample operator()(const DisneyMetal& m) const {
        return sample_disney_metal(wo, m, rng);
    }
    __device__ BsdfSample operator()(const DisneyClearcoat& m) const {
        return sample_disney_clearcoat(wo, m, rng);
    }
    __device__ BsdfSample operator()(const DisneyGlass& m) const {
        return sample_disney_glass(wo, m, rng);
    }
    __device__ BsdfSample operator()(const DisneySheen& m) const {
        return sample_disney_sheen(wo, m, rng);
    }
    __device__ BsdfSample operator()(const DisneyBsdf& m) const {
        return sample_disney_bsdf(wo, m, rng);
    }
};

__device__ BsdfEval eval_pdf_bsdf(glm::vec3 wo, glm::vec3 wi, const Material& m) {
    if (bx::cos_theta(wo) == 0.f) {
        return BsdfEval{};
    }
    return cuda::std::visit(EvalPdfOp{wo, wi}, m.bsdf);
}

__device__ BsdfSample sample_bsdf(glm::vec3 wo, const Material& m, RngEng& rng) {
    if (bx::cos_theta(wo) == 0.f) {
        return BsdfSample{};
    }
    return cuda::std::visit(SampleOp{wo, rng}, m.bsdf);
}
