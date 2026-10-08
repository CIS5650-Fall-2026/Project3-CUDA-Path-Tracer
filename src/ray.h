#pragma once

#include <glm/glm.hpp>

#include <cuda_runtime.h>

// Distance new rays are pushed off a surface so they don't immediately re-hit it
constexpr float RAY_OFFSET_EPSILON = 0.0001f;

struct Ray {
public:
	__host__ __device__ glm::vec3 getPositionAtTime(const float& t) const {
		return origin + t * direction;
	}

	__host__ __device__ Ray transform(const glm::mat4& transform) const {
		Ray transformed;
		transformed.origin = glm::vec3(transform * glm::vec4(origin, 1.0f));
		transformed.direction = glm::vec3(transform * glm::vec4(direction, 0.0f));
		return transformed;
	}

	// Construct new ray with origin shifted in direction of the normal as needed
	__host__ __device__ static Ray generateBouncedRay(const glm::vec3& normal, const glm::vec3& newOrigin, const glm::vec3& newDirection) {

		const float epsilon = copysignf(RAY_OFFSET_EPSILON, glm::dot(normal, newDirection));
		Ray newRay;
		newRay.origin = newOrigin + epsilon * normal;
		newRay.direction = newDirection;

		return newRay;
	}

	glm::vec3 origin = glm::vec3(0.0f);
	glm::vec3 direction = glm::vec3(1.0f, 0.0f, 0.0f);
};