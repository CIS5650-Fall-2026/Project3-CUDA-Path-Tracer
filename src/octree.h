#pragma once

#include "sceneStructs.h"

#include <vector>

#define OCTREE_MAX_DEPTH 8
#define OCTREE_LEAF_TRIS 16

// Traversal pops one node and pushes at most 8 children per level, so the
// stack never holds more than 7 * depth + 1 entries.
#define OCTREE_STACK_SIZE (7 * OCTREE_MAX_DEPTH + 8)

/**
 * Builds an octree over triangles[triStart, triStart + triCount) inside the
 * box [bboxMin, bboxMax] and appends it to `nodes` / `leafTriIndices`.
 * Triangles that span several octants are referenced by every leaf they touch.
 * Leaf entries in `leafTriIndices` are indices into `triangles`.
 *
 * @return                   Index of the root node in `nodes`.
 */
int buildMeshOctree(
    const std::vector<Triangle>& triangles,
    int triStart,
    int triCount,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax,
    std::vector<OctNode>& nodes,
    std::vector<int>& leafTriIndices);
