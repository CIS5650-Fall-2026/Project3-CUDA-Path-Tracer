#include "bvh.h"

#include "features.h"

#include <algorithm>
#include <cfloat>
#include <iostream>

using namespace std;

namespace
{
    // Same nudge as the mesh bounding box in scene.cpp, so that a face lying
    // exactly on a node's box can't be rejected by a grazing ray's rounding.
    // Every node gets the same padding, so a child's box still fits inside its
    // parent's.
    const float BOX_PADDING = 0.0001f;

    glm::vec3 centroid(const Triangle& tri)
    {
        return (tri.v0 + tri.v1 + tri.v2) * (1.0f / 3.0f);
    }

    struct Builder
    {
        vector<Triangle>& triangles;
        vector<BVHNode>& nodes;

        void computeBounds(BVHNode& node) const
        {
            glm::vec3 bboxMin(FLT_MAX);
            glm::vec3 bboxMax(-FLT_MAX);
            for (int i = node.leftOrFirst; i < node.leftOrFirst + node.triCount; ++i)
            {
                const Triangle& tri = triangles[i];
                bboxMin = glm::min(bboxMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
                bboxMax = glm::max(bboxMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
            }
            node.bboxMin = bboxMin - BOX_PADDING;
            node.bboxMax = bboxMax + BOX_PADDING;
        }

        // `node` must already hold its triangle range in leftOrFirst/triCount,
        // i.e. it starts life as a leaf and is turned into an internal node only
        // if it gets split.
        void subdivide(int nodeIdx, int depth)
        {
            // Copy rather than hold a reference: pushing the children below can
            // reallocate `nodes`.
            BVHNode node = nodes[nodeIdx];
            computeBounds(node);
            nodes[nodeIdx] = node;

            if (node.triCount <= BVH_LEAF_SIZE || depth >= BVH_MAX_DEPTH)
            {
                return;
            }

            int first = node.leftOrFirst;
            int count = node.triCount;

            // Split on the centroids rather than the node box: a few large
            // triangles can stretch the box along an axis the centroids don't
            // actually spread along.
            glm::vec3 centroidMin(FLT_MAX);
            glm::vec3 centroidMax(-FLT_MAX);
            for (int i = first; i < first + count; ++i)
            {
                glm::vec3 c = centroid(triangles[i]);
                centroidMin = glm::min(centroidMin, c);
                centroidMax = glm::max(centroidMax, c);
            }

            glm::vec3 extent = centroidMax - centroidMin;
            int axis = 0;
            if (extent.y > extent[axis]) axis = 1;
            if (extent.z > extent[axis]) axis = 2;

            if (extent[axis] <= 0.0f)
            {
                // Every centroid is the same point (stacked or degenerate
                // triangles). No plane can separate them, so stop here.
                return;
            }

            float splitPos = centroidMin[axis] + 0.5f * extent[axis];
            auto begin = triangles.begin() + first;
            auto end = begin + count;
            auto mid = partition(begin, end, [&](const Triangle& tri) {
                return centroid(tri)[axis] < splitPos;
            });
            int leftCount = static_cast<int>(mid - begin);

            if (leftCount == 0 || leftCount == count)
            {
                // The extent is so small that the midpoint rounded onto an end
                // of it. Split at the median instead, which always makes two
                // non-empty halves.
                leftCount = count / 2;
                nth_element(begin, begin + leftCount, end, [&](const Triangle& a, const Triangle& b) {
                    return centroid(a)[axis] < centroid(b)[axis];
                });
            }

            int leftIdx = static_cast<int>(nodes.size());
            nodes.push_back(BVHNode{ glm::vec3(0.0f), first, glm::vec3(0.0f), leftCount });
            nodes.push_back(BVHNode{ glm::vec3(0.0f), first + leftCount, glm::vec3(0.0f), count - leftCount });

            nodes[nodeIdx].leftOrFirst = leftIdx;
            nodes[nodeIdx].triCount = 0;

            subdivide(leftIdx, depth + 1);
            subdivide(leftIdx + 1, depth + 1);
        }
    };

    bool boxContains(const glm::vec3& outerMin, const glm::vec3& outerMax,
                     const glm::vec3& innerMin, const glm::vec3& innerMax)
    {
        return glm::all(glm::lessThanEqual(outerMin, innerMin))
            && glm::all(glm::lessThanEqual(innerMax, outerMax));
    }
}

int buildBVH(
    vector<Triangle>& triangles,
    int start,
    int count,
    vector<BVHNode>& nodes)
{
    int root = static_cast<int>(nodes.size());
    nodes.push_back(BVHNode{ glm::vec3(0.0f), start, glm::vec3(0.0f), count });

    Builder builder{ triangles, nodes };
    builder.subdivide(root, 0);
    return root;
}

bool validateBVH(
    const vector<Triangle>& triangles,
    int start,
    int count,
    const vector<BVHNode>& nodes,
    int root)
{
    vector<int> timesSeen(count, 0);
    int nodeCount = 0;
    int leafCount = 0;
    int maxDepth = 0;
    int maxLeafSize = 0;
    bool ok = true;

    auto fail = [&](const string& msg) {
        if (ok)
        {
            cout << "BVH validation failed: " << msg << endl;
        }
        ok = false;
    };

    struct Entry { int node; int depth; };
    vector<Entry> stack{ { root, 0 } };
    while (!stack.empty())
    {
        Entry e = stack.back();
        stack.pop_back();
        if (e.node < 0 || e.node >= static_cast<int>(nodes.size()))
        {
            fail("node index " + to_string(e.node) + " out of range");
            continue;
        }

        const BVHNode& node = nodes[e.node];
        ++nodeCount;
        maxDepth = max(maxDepth, e.depth);

        if (node.triCount > 0)
        {
            ++leafCount;
            maxLeafSize = max(maxLeafSize, node.triCount);
            if (e.depth > BVH_MAX_DEPTH)
            {
                fail("leaf " + to_string(e.node) + " deeper than BVH_MAX_DEPTH");
            }
            for (int i = node.leftOrFirst; i < node.leftOrFirst + node.triCount; ++i)
            {
                if (i < start || i >= start + count)
                {
                    fail("leaf " + to_string(e.node) + " reaches outside the mesh's triangles");
                    break;
                }
                ++timesSeen[i - start];
                const Triangle& tri = triangles[i];
                glm::vec3 triMin = glm::min(tri.v0, glm::min(tri.v1, tri.v2));
                glm::vec3 triMax = glm::max(tri.v0, glm::max(tri.v1, tri.v2));
                if (!boxContains(node.bboxMin, node.bboxMax, triMin, triMax))
                {
                    fail("triangle " + to_string(i) + " sticks out of leaf " + to_string(e.node));
                }
            }
            continue;
        }

        for (int child = node.leftOrFirst; child <= node.leftOrFirst + 1; ++child)
        {
            if (child <= e.node || child >= static_cast<int>(nodes.size()))
            {
                fail("node " + to_string(e.node) + " has bad child index " + to_string(child));
                continue;
            }
            if (!boxContains(node.bboxMin, node.bboxMax, nodes[child].bboxMin, nodes[child].bboxMax))
            {
                fail("child " + to_string(child) + " sticks out of node " + to_string(e.node));
            }
            stack.push_back({ child, e.depth + 1 });
        }
    }

    for (int i = 0; i < count; ++i)
    {
        if (timesSeen[i] != 1)
        {
            fail("triangle " + to_string(start + i) + " is in " + to_string(timesSeen[i]) + " leaves");
            break;
        }
    }

    cout << "BVH: " << nodeCount << " nodes, " << leafCount << " leaves, depth "
         << maxDepth << ", largest leaf " << maxLeafSize << " triangles" << endl;
    return ok;
}

void bvhUpload(
    const vector<Triangle>& triangles,
    const vector<BVHNode>& nodes,
    DeviceBVH& dev)
{
    dev.triangles = nullptr;
    dev.nodes = nullptr;

    if (!triangles.empty())
    {
        cudaMalloc(&dev.triangles, triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev.triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }
    if (!nodes.empty())
    {
        cudaMalloc(&dev.nodes, nodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev.nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }
}

void bvhFree(DeviceBVH& dev)
{
    cudaFree(dev.triangles);   // no-op on null
    cudaFree(dev.nodes);
    dev.triangles = nullptr;
    dev.nodes = nullptr;
}
