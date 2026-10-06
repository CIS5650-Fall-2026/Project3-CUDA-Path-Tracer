#pragma once

#include <cuda_runtime.h>
#include <OpenImageDenoise/oidn.hpp>
#include "glm/glm.hpp"

void initDenoiser(int width, int height, glm::vec3* d_image, glm::vec3* d_denoised);

void denoise();

void denoiserFree();



