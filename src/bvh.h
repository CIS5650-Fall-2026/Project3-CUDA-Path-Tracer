#pragma once

#include "sceneStructs.h"

#include <vector>

/**
 * Device copies of the scene's triangle and BVH node buffers. Each mesh Geom
 * indexes into these through triangleStart/triangleCount and bvhRoot.
 */
struct DeviceBVH
{
    Triangle* triangles;
    BVHNode* nodes;
};

/**
 * Builds a BVH over the triangles [start, start + count) of `triangles`,
 * appending its nodes to `nodes` and returning the root's index.
 *
 * The triangles are reordered in place, but only inside that range, so every
 * leaf covers a contiguous run of them and a mesh's triangleStart/triangleCount
 * stay valid. Node boxes live in the same (object) space as the triangles.
 *
 * Splits at the midpoint of the longest axis of the triangle centroids, stopping
 * at BVH_LEAF_SIZE triangles or BVH_MAX_DEPTH levels.
 */
int buildBVH(
    std::vector<Triangle>& triangles,
    int start,
    int count,
    std::vector<BVHNode>& nodes);

/**
 * Walks the BVH rooted at `root` and checks that it is a valid tree for the
 * triangles [start, start + count): every triangle sits in exactly one leaf,
 * every triangle is inside its leaf's box, every child box is inside its
 * parent's, and no leaf is deeper than BVH_MAX_DEPTH. Prints the tree's shape.
 *
 * @return  true if the tree passed every check.
 */
bool validateBVH(
    const std::vector<Triangle>& triangles,
    int start,
    int count,
    const std::vector<BVHNode>& nodes,
    int root);

/**
 * Allocates `dev` and copies both buffers to it. An empty buffer leaves its
 * pointer null.
 */
void bvhUpload(
    const std::vector<Triangle>& triangles,
    const std::vector<BVHNode>& nodes,
    DeviceBVH& dev);

/**
 * Frees what bvhUpload allocated and nulls the pointers, so it is safe to call
 * twice or on a DeviceBVH that was never uploaded.
 */
void bvhFree(DeviceBVH& dev);
