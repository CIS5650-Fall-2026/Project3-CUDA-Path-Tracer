#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

void setSDFScene(int i);
__host__ __device__ float sdfIntersectionTest(Ray r, glm::vec3& intersectionPoint, glm::vec3& normal);