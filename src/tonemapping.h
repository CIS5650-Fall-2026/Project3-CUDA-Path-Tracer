#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

namespace Tonemapping {
	__host__ __device__ inline glm::vec3 reinhardToneMap(const glm::vec3& color) {
		return color / (color + glm::vec3(1.0f));
	}

	__host__ __device__ inline glm::vec3 gammaCorrect(const glm::vec3& color, float gamma) {
		return glm::clamp(glm::pow(color, glm::vec3(1.0f / gamma)), 0.0f, 1.0f);
	}

	// Linear HDR radiance -> display ready [0, 1] color
	__host__ __device__ inline glm::vec3 toDisplayColor(const glm::vec3& hdrColor) {
		return gammaCorrect(reinhardToneMap(hdrColor), 2.2f);
	}
}
