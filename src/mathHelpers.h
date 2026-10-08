#pragma once

#include <cuda_runtime.h>

#include <cmath>

class MathHelpers {
public:
	__host__ __device__ static bool solveQuadratic(const float& a, const float& b, const float& c, float& x1, float& x2) {
		if (a == 0.0f) {
			return false;
		}

		float discriminant = b * b - 4.0f * a * c;

		if (discriminant < 0.0f) {
			return false;
		}

		float q = -0.5f * (b + copysignf(sqrtf(discriminant), b));

		x1 = q / a;
		x2 = c / q;

		return true;
	}

	__host__ __device__ static glm::vec3 safeNormalize(const glm::vec3 v) {
		float lengthSquared = glm::dot(v, v);
		if (lengthSquared > 0.0f) {
			return v * glm::inversesqrt(lengthSquared);
		}
		return glm::vec3(0.0f);
	}

	__host__ __device__ static void createCoordinateSystem(const glm::vec3& inNormal, glm::vec3& outTangent, glm::vec3& outBitangent) {
		float sign = std::copysign(1.0f, inNormal.z);
		const float a = -1.0f / (sign + inNormal.z);
		const float b = inNormal.x * inNormal.y * a;

		// Construct the Tangent
		outTangent = glm::vec3(1.0f + sign * inNormal.x * inNormal.x * a, sign * b, -sign * inNormal.x);

		// Construct the Bitangent
		outBitangent = glm::vec3(b, sign + inNormal.y * inNormal.y * a, -inNormal.y);
	}

	__host__ __device__ static bool isFinite(const glm::vec3& v) {
		return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
	}

	__host__ __device__ static bool isFinite(const glm::vec2& v) {
		return isfinite(v.x) && isfinite(v.y);
	}

	__host__ __device__ static float powerHeuristic(float pdfA, float pdfB) {
		return (pdfA * pdfA) / (pdfA * pdfA + pdfB * pdfB);
	}
};