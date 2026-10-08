#pragma once

#include <glm/glm.hpp>

#include "aabb.h"

#include "ray.h"

struct TLASNode {
	glm::vec3 aabbMin;
	int leftChild; // When == -1, this is a leaf node
	glm::vec3 aabbMax;
	int geometryIndex;

	__device__ bool intersect(const Ray& ray, float& tNearOut) const {
		// Using slab method for fast aabb
		glm::vec3 invDirection = 1.0f / ray.direction;

		glm::vec3 tMin = (aabbMin - ray.origin) * invDirection;
		glm::vec3 tMax = (aabbMax - ray.origin) * invDirection;

		glm::vec3 tNear = glm::min(tMin, tMax);
		glm::vec3 tFar = glm::max(tMin, tMax);

		float t0 = glm::max(tNear.x, glm::max(tNear.y, tNear.z));
		float t1 = glm::min(tFar.x, glm::min(tFar.y, tFar.z));

		tNearOut = (t0 < 0.0f) ? 0.0f : t0;

		return t0 <= t1 && t1 >= 0.0f;
	}
};

struct BLASNode {
	glm::vec3 aabbMin;
	int leftChild; // When == -1, this is a leaf node
	glm::vec3 aabbMax;
	int triangleOffset;
	int triangleCount;

	__device__ bool intersect(const Ray& ray, float& tNearOut) const {
		// Using slab method for fast aabb
		glm::vec3 invDirection = 1.0f / ray.direction;

		glm::vec3 tMin = (aabbMin - ray.origin) * invDirection;
		glm::vec3 tMax = (aabbMax - ray.origin) * invDirection;

		glm::vec3 tNear = glm::min(tMin, tMax);
		glm::vec3 tFar = glm::max(tMin, tMax);

		float t0 = glm::max(tNear.x, glm::max(tNear.y, tNear.z));
		float t1 = glm::min(tFar.x, glm::min(tFar.y, tFar.z));

		tNearOut = (t0 < 0.0f) ? 0.0f : t0;

		return t0 <= t1 && t1 >= 0.0f;
	}
};

struct GeometryBVHBuildData {
	int geometryIndex; // Index into dev_geometry buffer
	AABB bounds;
	glm::vec3 centroid;
};

struct TriangleBVHBuildData {
	int triangleIndex; // Index into dev_triangles buffer
	AABB bounds;
	glm::vec3 centroid;
};

class BVHBuilder {
public:
	static void buildBLAS(
		std::vector<TriangleBVHBuildData>& triangleData,
		int start, int end,
		std::vector<BLASNode>& outNodes,
		const std::vector<Triangle>& inTriangles,
		std::vector<Triangle>& outTriangles,
		int maxDepth = 20
	);

	static void buildTLAS(
		std::vector<GeometryBVHBuildData>& geometryData,
		int start, int end,
		std::vector<TLASNode>& outNodes,
		int maxDepth = 20
	);

private:
	static void buildBLASInternal(
		std::vector<TriangleBVHBuildData>& triangleData,
		int start, int end,
		int currentNodeIndex, // Node currently being filled
		std::vector<BLASNode>& outNodes,
		const std::vector<Triangle>& inTriangles,
		std::vector<Triangle>& outTriangles,
		int depth,
		int maxDepth
	);

	static void buildTLASInternal(
		std::vector<GeometryBVHBuildData>& geometryData,
		int start, int end,
		int currentNodeIndex, // Node currently being filled
		std::vector<TLASNode>& outNodes,
		int depth,
		int maxDepth
	);
};