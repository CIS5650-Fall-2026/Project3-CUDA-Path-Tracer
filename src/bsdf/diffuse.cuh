#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ BsdfEval eval_pdf_diffuse(glm::vec3 wi, const Lambertian& m) {
    float c = bx::non_neg_cos(wi);
    return BsdfEval{m.color * INV_PI * c, c * INV_PI};
}

__device__ __forceinline__ BsdfSample sample_diffuse(const Lambertian& m, RngEng& rng) {
    glm::vec3 wi = calculate_random_direction_in_cosine_hemisphere(rng);
    BsdfEval e = eval_pdf_diffuse(wi, m);
    return {wi, e.pdf, e.f, BxdfFlag::Diffuse};
}
