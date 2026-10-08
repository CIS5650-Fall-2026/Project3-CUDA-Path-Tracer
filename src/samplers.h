#pragma once

#include <cuda_runtime.h>

#include <glm/gtc/constants.hpp>

#include "mathHelpers.h"

class Samplers {
public:
	__host__ __device__ static glm::vec2 sampleUniformDiskConcentric(const glm::vec2& random) {
		glm::vec2 uOffset = 2.0f * random - glm::vec2(1.f);
		if (uOffset.x == 0.0f && uOffset.y == 0.0f) {
			return glm::vec2(0.0f);
		}

		float theta;
		float r;
		if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
			r = uOffset.x;
			theta = (glm::pi<float>() / 4.0f) * (uOffset.y / uOffset.x);
		}
		else {
			r = uOffset.y;
			theta = (glm::pi<float>() / 2.0f) - (glm::pi<float>() / 4.0f) * (uOffset.x / uOffset.y);
		}
		return r * glm::vec2(glm::cos(theta), glm::sin(theta));
	}

	__host__ __device__ static glm::vec3 sampleCosineWeightedHemisphere(const glm::vec3& normal, const glm::vec2& random, float& outPdf) {
		glm::vec3 localSample = sampleLocalCosineWeightedHemisphere(random);

		glm::vec3 tangent;
		glm::vec3 bitangent;
		MathHelpers::createCoordinateSystem(normal, tangent, bitangent);

		glm::vec3 outgoingDirection = localSample.x * tangent + localSample.y * bitangent + localSample.z * normal;

		float cosTheta = glm::max(0.0f, glm::dot(normal, outgoingDirection));
		outPdf = cosTheta / glm::pi<float>();

		return outgoingDirection;
	}

	__host__ __device__ static glm::vec3 sampleGeometry(const Geom& geometry, const Triangle* dev_triangles, const glm::vec3& random, glm::vec3& outNormal, float& outPdf) {
		glm::vec3 localSamplePoint = glm::vec3(0.0f);
		glm::vec3 localSampleNormal = glm::vec3(0.0f);
		float localPdf = 1.0f;
		if (geometry.type == GeomType::CUBE) {
			localSamplePoint = sampleUnitCube(random, localSampleNormal, localPdf);
		}
		else if (geometry.type == GeomType::SPHERE) {
			localSamplePoint = sampleUnitSphere(glm::vec2(random.x, random.y), localSampleNormal, localPdf);
		}
		else if (geometry.type == GeomType::MESH) {
			localSamplePoint = sampleMesh(geometry, dev_triangles, random, localSampleNormal, localPdf);
		}
		else {
			outPdf = 0.0f;
			return glm::vec3(0.0f);
		}

		glm::vec3 worldNormal = glm::vec3(geometry.invTranspose * glm::vec4(localSampleNormal, 0.0f));

		float normalScale = glm::length(worldNormal);

		outNormal = worldNormal / normalScale;

		float det = glm::abs(glm::determinant(geometry.transform));
		float jacobian = det * normalScale;

		if (jacobian <= 0.0f) {
			outPdf = 0.0f;
		}
		else {
			outPdf = localPdf / jacobian;
		}

		return glm::vec3(geometry.transform * glm::vec4(localSamplePoint, 1.0f));
	}

private:
	__host__ __device__ static glm::vec3 sampleLocalCosineWeightedHemisphere(const glm::vec2& random) {
		glm::vec2 d = sampleUniformDiskConcentric(random);
		float z = glm::sqrt(1 - d.x * d.x - d.y * d.y);
		return glm::vec3(d.x, d.y, z);
	}

	__host__ __device__ static glm::vec3 sampleUnitSphere(const glm::vec2& random, glm::vec3& outNormal, float& outPdf) {
		float z = 1.0 - 2.0 * random.x;

		float phi = 2.0 * glm::pi<float>() * random.y;

		float r = glm::sqrt(glm::max(0.0f, 1.0f - z * z));

		outNormal.x = r * glm::cos(phi);
		outNormal.y = r * glm::sin(phi);
		outNormal.z = z;

		outPdf = 1.0 / glm::pi<float>();

		return 0.5f * outNormal;
	}

	__host__ __device__ static glm::vec3 sampleUnitCube(const glm::vec3& random, glm::vec3& outNormal, float& outPdf) {
		int face = (int)(random.x * 6.0);

		float u = random.y - 0.5f;
		float v = random.z - 0.5f;

		// TODO: Consider a branchless version
		outPdf = 1.0f / 6.0f;

		if (face == 0) {
			outNormal = glm::vec3(1.0, 0.0, 0.0);
			return glm::vec3(0.5, u, v);
		}
		else if (face == 1) {
			outNormal = glm::vec3(-1.0, 0.0, 0.0);
			return glm::vec3(-0.5, u, v);
		}
		else if (face == 2) {
			outNormal = glm::vec3(0.0, 1.0, 0.0);
			return glm::vec3(u, 0.5, v);
		}
		else if (face == 3) {
			outNormal = glm::vec3(0.0, -1.0, 0.0);
			return glm::vec3(u, -0.5, v);
		}
		else if (face == 4) {
			outNormal = glm::vec3(0.0, 0.0, 1.0);
			return glm::vec3(u, v, 0.5);
		}
		else {
			outNormal = glm::vec3(0.0, 0.0, -1.0);
			return glm::vec3(u, v, -0.5);
		}
	}

	__host__ __device__ static glm::vec3 sampleMesh(
		const Geom& geometry,
		const Triangle* dev_triangles,
		const glm::vec3& random,
		glm::vec3& outNormal,
		float& outPdf
	) {
		if(geometry.triangleCount <= 0) {
			outPdf = 0.0f;
			outNormal = glm::vec3(0.0f, 1.0f, 0.0f);
			return glm::vec3(0.0f);
		}

		// Pick triangle at random (not weighted by surface area)
		int triangleIndex = (int)(random.x * geometry.triangleCount);
		const Triangle& tri = dev_triangles[geometry.triangleOffset + triangleIndex];

		// Sample using barycentric coordinates
		float u = random.y;
		float v = random.z;
		if (u + v > 1.0f) {
			u = 1.0f - u;
			v = 1.0f - v;
		}
		float w = 1.0f - u - v;

		glm::vec3 localSamplePoint = u * tri.v0 + v * tri.v1 + w * tri.v2;

		// Get object space normal
		glm::vec3 edge1 = tri.v1 - tri.v0;
		glm::vec3 edge2 = tri.v2 - tri.v0;
		outNormal = MathHelpers::safeNormalize(glm::cross(edge1, edge2));

		// Calculate pdf
		float triangleArea = 0.5f * glm::length(glm::cross(edge1, edge2));
		if(triangleArea <= 0.0f) {
			outPdf = 0.0f;
		} else {
			// pdf is 1 / (area * num triangles)
			outPdf = 1.0f / (triangleArea * geometry.triangleCount);
		}

		return localSamplePoint;
	}
};