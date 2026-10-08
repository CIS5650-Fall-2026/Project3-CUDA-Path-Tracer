#include "bvh.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <iostream>

namespace {
    struct Bounds {
        glm::vec3 min = glm::vec3(FLT_MAX);
        glm::vec3 max = glm::vec3(-FLT_MAX);

        void grow(const glm::vec3& p) {
            min = glm::min(min, p);
            max = glm::max(max, p);
        }

        void grow(const Bounds& b) {
            min = glm::min(min, b.min);
            max = glm::max(max, b.max);
        }

        float halfArea() const {
            const glm::vec3 e = max - min;
            return e.x * e.y + e.y * e.z + e.z * e.x;
        }
    };

    struct Builder {
        std::vector<BVHNode>& nodes;
        int firstTriangle;
        std::vector<Bounds> triangleBounds;
        std::vector<glm::vec3> centroids;
        std::vector<int> order;

        int deepestLeaf = 0;
        int leafCount = 0;
        int largestLeaf = 0;

        void subdivide(int nodeIndex, int begin, int end, int depth) {
            Bounds bounds;
            Bounds centroidBounds;
            for (int i = begin; i < end; i++) {
                bounds.grow(triangleBounds[order[i]]);
                centroidBounds.grow(centroids[order[i]]);
            }
            nodes[nodeIndex].boundsMin = bounds.min;
            nodes[nodeIndex].boundsMax = bounds.max;

            const int count = end - begin;
            int bestAxis = -1;
            int bestBin = 0;
            float bestCost = FLT_MAX;

            // find the best split
            if (count > BVH_MAX_LEAF_SIZE && depth < BVH_MAX_DEPTH) {
                for (int axis = 0; axis < 3; axis++) {
                    const float extent = centroidBounds.max[axis] - centroidBounds.min[axis];
                    if (!(extent > 0.0f)) {
                        continue;
                    }

                    Bounds binBounds[BVH_SAH_BINS];
                    int binCount[BVH_SAH_BINS] = {};
                    const float scale = BVH_SAH_BINS / extent;
                    for (int i = begin; i < end; i++) {
                        const int t = order[i];
                        const int bin = std::min(BVH_SAH_BINS - 1, (int)((centroids[t][axis] - centroidBounds.min[axis]) * scale));
                        binBounds[bin].grow(triangleBounds[t]);
                        binCount[bin]++;
                    }

                    float rightArea[BVH_SAH_BINS - 1];
                    int rightCount[BVH_SAH_BINS - 1];
                    Bounds swept;
                    int sweptCount = 0;
                    for (int plane = BVH_SAH_BINS - 2; plane >= 0; plane--) {
                        swept.grow(binBounds[plane + 1]);
                        sweptCount += binCount[plane + 1];
                        rightArea[plane] = swept.halfArea();
                        rightCount[plane] = sweptCount;
                    }

                    swept = Bounds();
                    sweptCount = 0;
                    for (int plane = 0; plane < BVH_SAH_BINS - 1; plane++) {
                        swept.grow(binBounds[plane]);
                        sweptCount += binCount[plane];
                        if (sweptCount == 0 || rightCount[plane] == 0) {
                            continue;
                        }
                        const float cost = swept.halfArea() * sweptCount + rightArea[plane] * rightCount[plane];
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestAxis = axis;
                            bestBin = plane;
                        }
                    }
                } // first for-loop
            } // if

            //make a leaf or split
            if (bestAxis < 0) {
                nodes[nodeIndex].leftOrFirst = firstTriangle + begin;
                nodes[nodeIndex].triangleCount = count;
                deepestLeaf = std::max(deepestLeaf, depth);
                largestLeaf = std::max(largestLeaf, count);
                leafCount++;
                return;
            }

            const float origin = centroidBounds.min[bestAxis];
            const float scale = BVH_SAH_BINS / (centroidBounds.max[bestAxis] - origin);
            const int mid = (int)(std::partition(order.begin() + begin, order.begin() + end,
                [&](int t) {
                    const int bin = std::min(BVH_SAH_BINS - 1, (int)((centroids[t][bestAxis] - origin) * scale));
                    return bin <= bestBin;
                }) - order.begin());

            const int left = (int)nodes.size();
            nodes.emplace_back();
            nodes.emplace_back();
            nodes[nodeIndex].leftOrFirst = left;
            nodes[nodeIndex].triangleCount = 0;
            subdivide(left, begin, mid, depth + 1);
            subdivide(left + 1, mid, end, depth + 1);
        }
    };
}

int buildBVH(std::vector<Triangle>& triangles, int first, int count, std::vector<BVHNode>& nodes) {
    if (count <= 0) {
        return -1;
    }

    const auto start = std::chrono::steady_clock::now();

    Builder builder{ nodes, first };
    builder.triangleBounds.resize(count);
    builder.centroids.resize(count);
    builder.order.resize(count);
    for (int i = 0; i < count; i++) {
        const Triangle& tri = triangles[first + i];
        builder.triangleBounds[i].grow(tri.v0);
        builder.triangleBounds[i].grow(tri.v1);
        builder.triangleBounds[i].grow(tri.v2);
        builder.centroids[i] = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
        builder.order[i] = i;
    }

    const int root = (int)nodes.size();
    nodes.emplace_back();
    builder.subdivide(root, 0, count, 0);

    std::vector<Triangle> ordered(count);
    for (int i = 0; i < count; i++) {
        ordered[i] = triangles[first + builder.order[i]];
    }
    std::copy(ordered.begin(), ordered.end(), triangles.begin() + first);

    const double miliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
        std::cout << "Built BVH in " << miliseconds << " ms: " << nodes.size() - root << " nodes, "
        << builder.leafCount << " leaves averaging " << (float)count / builder.leafCount
        << " triangles (largest " << builder.largestLeaf << "), depth " << builder.deepestLeaf << std::endl;
    return root;
}
