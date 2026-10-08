#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

__host__ __device__ glm::mat3 FrameFromZ(glm::vec3 v1);
__host__ __device__ glm::vec3 SphericalDirection(float cosTheta, float sinTheta, float phi);