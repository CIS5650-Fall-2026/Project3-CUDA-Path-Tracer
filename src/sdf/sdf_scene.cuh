#pragma once

#include "../scene_structs.h"

#include <cuda_runtime.h>

__device__ float scene_intersect(Ray r, float t_min, float t_max, float eps);

__device__ glm::vec3 scene_normal(glm::vec3 p, float eps);
