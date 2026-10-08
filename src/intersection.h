#pragma once

#include "ray.h"
#include "mathHelpers.h"
#include "bvh.h"
#include "devScene.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

struct IntersectionData {
	glm::vec3 normal = glm::vec3(1, 0, 0);
	float t = -1.0f;
	glm::vec2 uv = glm::vec2(0.0f);
	int geometryIndex = -1;
	int materialIndex = 0;
	bool bInside = false;
	float pdfIfLight = 0.0f;

	// BVH iteration counts for debug
	int blasIterationCount = 0;
	int tlasIterationCount = 0;
};

class IntersectionStatics {
public:
	__device__ static IntersectionData intersectGeometry(
		const Ray& ray, 
		const Geom& geometry, 
		const BLASNode* dev_blasNodes, 
		const Triangle* dev_triangles, 
		int lightCount);

private:
	__device__ static IntersectionData intersectSphere(const Ray& ray);

	__device__ static IntersectionData intersectBox(const Ray& ray);

	__device__ static IntersectionData intersectTriangle(const Ray& ray, const Triangle& tri);

	__device__ static IntersectionData intersectMesh(
		const Ray& ray,
		const Geom& geometry,
		const BLASNode* blasNodes,
		const Triangle* sceneTriangles);
};
