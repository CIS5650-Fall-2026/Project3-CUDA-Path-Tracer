#include "aabb.h"

#include "bvh.h"

#include <stdexcept>

void AABB::expand(const glm::vec3& point) {
	min = glm::min(min, point);
	max = glm::max(max, point);
}

void AABB::expand(const AABB& other) {
	min = glm::min(min, other.min);
	max = glm::max(max, other.max);
}

AABB AABB::fromTriangle(const Triangle& triangle) {
	AABB box;
	box.expand(triangle.v0);
	box.expand(triangle.v1);
	box.expand(triangle.v2);
	return box;
}

AABB AABB::fromGeometry(const Geom& geometry, const std::vector<BLASNode>& blasNodes) {
	switch (geometry.type)
	{
	case GeomType::SPHERE:
		return fromSphere(geometry.transform);
	case GeomType::CUBE:
		return fromCube(geometry.transform);
	case GeomType::MESH:
		return fromBlasNode(blasNodes[geometry.blasNodeOffset], geometry.transform);
	default:
		throw std::runtime_error("Unknown geometry type for AABB generation");
	}
}

AABB AABB::fromSphere(const glm::mat4& transform) {
	AABB box;

	glm::vec3 center = glm::vec3(transform[3]);

	glm::vec3 r;
	r.x = glm::length(glm::vec3(transform[0][0], transform[1][0], transform[2][0])) * 0.5f;
	r.y = glm::length(glm::vec3(transform[0][1], transform[1][1], transform[2][1])) * 0.5f;
	r.z = glm::length(glm::vec3(transform[0][2], transform[1][2], transform[2][2])) * 0.5f;

	box.min = center - r;
	box.max = center + r;

	return box;
}

AABB AABB::fromCube(const glm::mat4& transform) {
	AABB box;

	glm::vec3 center = glm::vec3(transform[3]);

	glm::vec3 right = glm::vec3(transform[0]);
	glm::vec3 up = glm::vec3(transform[1]);
	glm::vec3 forward = glm::vec3(transform[2]);

	glm::vec3 halfExtents = (glm::abs(right) + glm::abs(up) + glm::abs(forward)) * 0.5f;

	box.min = center - halfExtents;
	box.max = center + halfExtents;

	return box;
}

AABB AABB::fromBlasNode(const BLASNode& node, const glm::mat4& transform) {
	AABB box;

	glm::vec3 localCenter = (node.aabbMin + node.aabbMax) * 0.5f;
	glm::vec3 localHalfExtents = (node.aabbMax - node.aabbMin) * 0.5f;

	glm::vec3 worldCenter = glm::vec3(transform * glm::vec4(localCenter, 1.0f));

	glm::vec3 right = glm::vec3(transform[0]);
	glm::vec3 up = glm::vec3(transform[1]);
	glm::vec3 forward = glm::vec3(transform[2]);

	glm::vec3 worldHalfExtents = glm::abs(right) * localHalfExtents.x +
		glm::abs(up) * localHalfExtents.y +
		glm::abs(forward) * localHalfExtents.z;

	box.min = worldCenter - worldHalfExtents;
	box.max = worldCenter + worldHalfExtents;

	return box;
}