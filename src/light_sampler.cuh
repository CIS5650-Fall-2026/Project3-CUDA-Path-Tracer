#pragma once

#include "scene_structs.h"

struct LightSampler {
    const Light* lights;
    int num_lights;
    cudaTextureObject_t env_texture;
    float p_env;
};

__device__ inline float pmf_env(const LightSampler& s)  { return s.p_env; }
__device__ inline float pmf_area(const LightSampler& s) {
    return s.num_lights > 0 ? (1.f - s.p_env) / s.num_lights : 0.f;
}
