#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <limits>

#include "pathTraceCommon.h"

struct BLASNode;

struct AABB {
	glm::vec3 min = glm::vec3((std::numeric_limits<float>::infinity)());
	glm::vec3 max = glm::vec3(-(std::numeric_limits<float>::infinity)());

	void expand(const glm::vec3& point);

	void expand(const AABB& other);

	static AABB fromTriangle(const Triangle& triangle);

	static AABB fromGeometry(const Geom& geometry, const std::vector<BLASNode>& blasNodes);

private:
	static AABB fromSphere(const glm::mat4& transform);

	static AABB fromCube(const glm::mat4& transform);

	static AABB fromBlasNode(const BLASNode& node, const glm::mat4& transform);
};