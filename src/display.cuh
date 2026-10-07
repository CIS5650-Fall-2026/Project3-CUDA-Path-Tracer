#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>


__host__ __device__ glm::ivec3 to_display(glm::vec3 linear_rgb, bool agx_tonemap);

__global__ void send_image_to_pbo(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool agx_tonemap);
