#pragma once

#include "../scene_structs.h"
#include "../thrust_utils.h"
#include <glm/glm.hpp>

__device__ BsdfEval eval_pdf_bsdf(glm::vec3 wo, glm::vec3 wi, const Material& m);

__device__ BsdfSample sample_bsdf(glm::vec3 wo, const Material& m, RngEng& rng);

__device__ __forceinline__ bool is_delta(const Material& m) {
    return cuda::std::visit([](const auto& b) { return b.is_delta(); }, m.bsdf);
}

__device__ __forceinline__ bool is_emissive(const Material& m) {
    return m.emission.strength > 0.f;
}
