#include "octree.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace
{
    struct TriBounds
    {
        glm::vec3 lo;
        glm::vec3 hi;
    };

    struct BuildStats
    {
        int leaves = 0;
        int emptyLeaves = 0;
        int maxDepth = 0;
        int maxLeafTris = 0;
        size_t triRefs = 0;
    };

    struct BuildContext
    {
        const std::vector<Triangle>& triangles;
        const std::vector<TriBounds>& bounds; // indexed by triangle index - triStart
        int triStart;
        std::vector<OctNode>& nodes;
        std::vector<int>& leafTriIndices;
        BuildStats stats;
    };

    bool boxesOverlap(const glm::vec3& aLo, const glm::vec3& aHi,
        const glm::vec3& bLo, const glm::vec3& bHi)
    {
        return aLo.x <= bHi.x && aHi.x >= bLo.x
            && aLo.y <= bHi.y && aHi.y >= bLo.y
            && aLo.z <= bHi.z && aHi.z >= bLo.z;
    }

    // Separating axis test (Akenine-Moller): box axes, triangle normal, and the
    // 9 cross products of box axes with triangle edges.
    bool triangleBoxOverlap(const Triangle& tri, const glm::vec3& boxLo, const glm::vec3& boxHi)
    {
        glm::vec3 c = 0.5f * (boxLo + boxHi);
        glm::vec3 h = 0.5f * (boxHi - boxLo);
        glm::vec3 v[3] = { tri.v0 - c, tri.v1 - c, tri.v2 - c };
        glm::vec3 e[3] = { v[1] - v[0], v[2] - v[1], v[0] - v[2] };

        for (int i = 0; i < 3; ++i)
        {
            for (int a = 0; a < 3; ++a)
            {
                glm::vec3 axisDir(0.0f);
                axisDir[a] = 1.0f;
                glm::vec3 axis = glm::cross(axisDir, e[i]);
                float p0 = glm::dot(v[0], axis);
                float p1 = glm::dot(v[1], axis);
                float p2 = glm::dot(v[2], axis);
                float r = h.x * fabsf(axis.x) + h.y * fabsf(axis.y) + h.z * fabsf(axis.z);
                if (std::min(p0, std::min(p1, p2)) > r || std::max(p0, std::max(p1, p2)) < -r)
                {
                    return false;
                }
            }
        }

        for (int a = 0; a < 3; ++a)
        {
            float lo = std::min(v[0][a], std::min(v[1][a], v[2][a]));
            float hi = std::max(v[0][a], std::max(v[1][a], v[2][a]));
            if (lo > h[a] || hi < -h[a])
            {
                return false;
            }
        }

        glm::vec3 n = glm::cross(e[0], e[1]);
        float d = glm::dot(n, v[0]);
        float r = h.x * fabsf(n.x) + h.y * fabsf(n.y) + h.z * fabsf(n.z);
        return fabsf(d) <= r;
    }

    void makeLeaf(BuildContext& ctx, int nodeIndex, const std::vector<int>& tris)
    {
        OctNode& node = ctx.nodes[nodeIndex];
        node.firstChild = -1;
        node.triStart = (int)ctx.leafTriIndices.size();
        node.triCount = (int)tris.size();
        ctx.leafTriIndices.insert(ctx.leafTriIndices.end(), tris.begin(), tris.end());

        ctx.stats.leaves++;
        if (tris.empty())
        {
            ctx.stats.emptyLeaves++;
        }
        ctx.stats.maxLeafTris = std::max(ctx.stats.maxLeafTris, (int)tris.size());
        ctx.stats.triRefs += tris.size();
    }

    void buildNode(BuildContext& ctx, int nodeIndex, const std::vector<int>& tris, int depth)
    {
        ctx.stats.maxDepth = std::max(ctx.stats.maxDepth, depth);
        if ((int)tris.size() <= OCTREE_LEAF_TRIS || depth >= OCTREE_MAX_DEPTH)
        {
            makeLeaf(ctx, nodeIndex, tris);
            return;
        }

        glm::vec3 lo = ctx.nodes[nodeIndex].bboxMin;
        glm::vec3 hi = ctx.nodes[nodeIndex].bboxMax;
        glm::vec3 mid = 0.5f * (lo + hi);

        glm::vec3 childLo[8];
        glm::vec3 childHi[8];
        std::vector<int> childTris[8];
        for (int c = 0; c < 8; ++c)
        {
            childLo[c] = glm::vec3((c & 1) ? mid.x : lo.x, (c & 2) ? mid.y : lo.y, (c & 4) ? mid.z : lo.z);
            childHi[c] = glm::vec3((c & 1) ? hi.x : mid.x, (c & 2) ? hi.y : mid.y, (c & 4) ? hi.z : mid.z);
        }

        for (int tri : tris)
        {
            const TriBounds& b = ctx.bounds[tri - ctx.triStart];
            for (int c = 0; c < 8; ++c)
            {
                if (boxesOverlap(b.lo, b.hi, childLo[c], childHi[c])
                    && triangleBoxOverlap(ctx.triangles[tri], childLo[c], childHi[c]))
                {
                    childTris[c].push_back(tri);
                }
            }
        }

        // Splitting is pointless if some octant still holds every triangle.
        for (int c = 0; c < 8; ++c)
        {
            if (childTris[c].size() == tris.size())
            {
                makeLeaf(ctx, nodeIndex, tris);
                return;
            }
        }

        int firstChild = (int)ctx.nodes.size();
        ctx.nodes.resize(firstChild + 8);
        ctx.nodes[nodeIndex].firstChild = firstChild;
        ctx.nodes[nodeIndex].triStart = 0;
        ctx.nodes[nodeIndex].triCount = 0;

        for (int c = 0; c < 8; ++c)
        {
            ctx.nodes[firstChild + c].bboxMin = childLo[c];
            ctx.nodes[firstChild + c].bboxMax = childHi[c];
            buildNode(ctx, firstChild + c, childTris[c], depth + 1);
        }
    }
}

int buildMeshOctree(
    const std::vector<Triangle>& triangles,
    int triStart,
    int triCount,
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax,
    std::vector<OctNode>& nodes,
    std::vector<int>& leafTriIndices)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<TriBounds> bounds(triCount);
    std::vector<int> rootTris(triCount);
    for (int i = 0; i < triCount; ++i)
    {
        const Triangle& t = triangles[triStart + i];
        bounds[i].lo = glm::min(t.v0, glm::min(t.v1, t.v2));
        bounds[i].hi = glm::max(t.v0, glm::max(t.v1, t.v2));
        rootTris[i] = triStart + i;
    }

    size_t nodesBefore = nodes.size();
    int root = (int)nodes.size();
    nodes.emplace_back();
    nodes[root].bboxMin = bboxMin;
    nodes[root].bboxMax = bboxMax;

    BuildContext ctx{ triangles, bounds, triStart, nodes, leafTriIndices, BuildStats{} };
    buildNode(ctx, root, rootTris, 0);

    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - startTime).count();

    const BuildStats& s = ctx.stats;
    int filledLeaves = s.leaves - s.emptyLeaves;
    std::cout << "Octree: " << (nodes.size() - nodesBefore) << " nodes, "
        << s.leaves << " leaves (" << s.emptyLeaves << " empty), "
        << "max depth " << s.maxDepth << ", "
        << "avg " << (filledLeaves > 0 ? (double)s.triRefs / filledLeaves : 0.0)
        << " / max " << s.maxLeafTris << " tris per leaf, "
        << s.triRefs << " tri refs (" << (triCount > 0 ? (double)s.triRefs / triCount : 0.0)
        << "x duplication), built in " << ms << " ms" << std::endl;

    return root;
}
