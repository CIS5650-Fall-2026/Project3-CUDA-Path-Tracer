#pragma once

#include "sceneStructs.h"

#include <vector>

struct BVH
{
    std::vector<BVHNode> nodes;
    BVHNode* dev_nodes = NULL;
    int maxDepth = 32;
    int leafSize = 4;

    int build(std::vector<Triangle>& triangles, int start, int end, int depth = 1);
    void initDevice();
    void freeDevice();
};
