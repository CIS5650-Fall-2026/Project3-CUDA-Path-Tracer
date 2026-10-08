#pragma once

#include <glm/glm.hpp>

#include "ray.h"

struct PathState {
	Ray ray = Ray{};
	glm::vec3 throughput = glm::vec3(1.0f);
	glm::vec3 accumulatedColor = glm::vec3(0.0f);
	glm::vec3 directColor = glm::vec3(0.0f); // Portion of accumulatedColor that is from direct lighting
	int bounceCount = 0;
	bool active = true;
	int pixelIndex = -1;

	// For MIS
	float previousBrdfPdf = 0.0f;
	bool previousDelta = true;
};

enum GeomType
{
	SPHERE,
	CUBE,
	MESH
};

struct Triangle {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
	glm::vec2 uv0;
	glm::vec2 uv1;
	glm::vec2 uv2;
};

struct Geom
{
	enum GeomType type = GeomType::SPHERE;
	int materialid = 0;

	// TODO: Remove vec3s from device since these should be stored in the mat4s and are redundant
	glm::vec3 translation = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);

	glm::mat4 transform = glm::mat4(1.0f);
	glm::mat4 inverseTransform = glm::mat4(1.0f);
	glm::mat4 invTranspose = glm::mat4(1.0f);

	int blasNodeOffset = 0;
	int triangleOffset = 0;
	int triangleCount = 0;
};