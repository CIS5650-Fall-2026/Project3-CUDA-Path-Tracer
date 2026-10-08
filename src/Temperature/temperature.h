#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>

__device__ glm::vec3 kelvin_to_rgb(float temperature);