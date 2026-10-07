#pragma once
#include "light_sampler.cuh"
#include "scene_structs.h"
#include "thrust_utils.h"
#include <cuda/std/optional>
#include <cuda_runtime.h>

namespace cstd = cuda::std;

__device__ cstd::optional<LightSample> sample_direct_light(glm::vec3 p, glm::vec3 nor,
                                                           const LightSampler& s,
                                                           const Geom* geoms, RngEng& rng);

__device__ float pdf_area_light(const Ray& r, const Light& light, const Geom* geoms);

__device__ float pdf_env_light(glm::vec3 wi);
