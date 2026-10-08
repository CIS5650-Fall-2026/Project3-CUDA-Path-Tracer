#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include <vector>
#include <string>

// Environment map data for CUDA similar to DevScene
struct DevEnvironmentMap {
	__device__ glm::vec3 sample(const glm::vec3& direction) const;

	cudaTextureObject_t texture = 0;
	float intensity = 1.0f;
};

class EnvironmentMap {
public:
	EnvironmentMap(float* imageData, size_t width, size_t height);
	EnvironmentMap(const std::string& filepath);
	~EnvironmentMap();

	EnvironmentMap& operator=(const EnvironmentMap&) = delete;
	EnvironmentMap(const EnvironmentMap&) = delete;

	EnvironmentMap& operator=(EnvironmentMap&&) noexcept;
	EnvironmentMap(EnvironmentMap&&) noexcept;

	DevEnvironmentMap getDevEnvironmentMap() const {
		return DevEnvironmentMap{ m_environmentMapTexture, m_intensity };
	}

	cudaTextureObject_t m_environmentMapTexture = 0;

	// Scales the radiance from the map
	float m_intensity = 1.0f;

private:
	cudaArray_t m_environmentMapArray = nullptr;
};
