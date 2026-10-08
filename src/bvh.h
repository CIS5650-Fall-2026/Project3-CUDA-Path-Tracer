#pragma once

#include "sceneStructs.h"

#include <vector>

#define BVH_MAX_DEPTH 32
#define BVH_MAX_LEAF_SIZE 4
#define BVH_SAH_BINS 16

struct BVHNode {
    glm::vec3 boundsMin;
    int leftOrFirst;
    glm::vec3 boundsMax;
    int triangleCount;
};

int buildBVH(std::vector<Triangle>& triangles, int first, int count, std::vector<BVHNode>& nodes);