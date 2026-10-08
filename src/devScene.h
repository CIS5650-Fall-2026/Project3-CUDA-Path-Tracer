#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Ray;
struct IntersectionData;

struct Geom;
struct Triangle;
struct BLASNode;
struct TLASNode;
struct Material;

struct DevScene {
	__device__ IntersectionData intersect(const Ray& ray);

	__device__ glm::vec3 nextEventEstimation(const glm::vec4& random, const Ray& incomingRay, const IntersectionData& intersectionData, glm::vec3& outDirectionToLight, float& outPdf);

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;

	Triangle* dev_triangles = nullptr;
	size_t m_triangleCount = 0;

	BLASNode* dev_blasNodes = nullptr;
	size_t m_blasNodeCount = 0;

	TLASNode* dev_tlasNodes = nullptr;
	size_t m_tlasNodeCount = 0;

	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;

	size_t m_lightCount = 0;

private:
	__device__ bool isVisible(const Ray& ray, float tMax);
};