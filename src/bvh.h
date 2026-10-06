#pragma once

#include "sceneStructs.h"

#include <memory>
#include <vector>

class Bounds3 {
public:
    glm::vec3 pMin;
    glm::vec3 pMax;

    Bounds3();
    bool IsEmpty() const;
    void Expand(const glm::vec3& point);
    void Expand(const Bounds3& bounds);
    float SurfaceArea() const;
    int MaxDimension() const;
};

class BVHPrimitive {
public:
    int primitiveIndex;
    Bounds3 bounds;

    BVHPrimitive(int primitiveIndex, const Bounds3& bounds);
    glm::vec3 Centroid() const;
};

class BVHBuildNode {
public:
    Bounds3 bounds;
    BVHBuildNode* children[2] = {nullptr, nullptr};
    int splitAxis = 0;
    int firstPrimOffset = 0;
    int nPrimitives = 0;

    void InitLeaf(int first, int count, const Bounds3& bounds);
    void InitInterior(int axis, BVHBuildNode* child0, BVHBuildNode* child1);
};

class BVHSplitBucket {
public:
    int count = 0;
    Bounds3 bounds;
};

class LinearBVHNode {
public:
    Bounds3 bounds;
    int primitivesOffset;
    int secondChildOffset;
    int nPrimitives;
    int axis;
};

class BVH {
public:
    BVH(const std::vector<Geom>& primitives, int maxPrimsInNode, int maxDepth);
    void Build(std::vector<LinearBVHNode>& nodes, std::vector<int>& orderedPrimitives);

private:
    const std::vector<Geom>& primitives;
    int maxPrimsInNode;
    int maxDepth; // default 64 for now
    std::vector<std::unique_ptr<BVHBuildNode>> allocatedNodes;
    std::vector<int> orderedPrimitives;

    Bounds3 PrimitiveBounds(const Geom& primitive) const;
    BVHBuildNode* AllocateNode();
    BVHBuildNode* CreateLeaf(BVHBuildNode* node, const std::vector<BVHPrimitive>& bvhPrimitives, int start, int end, const Bounds3& bounds);
    BVHBuildNode* BuildRecursive(std::vector<BVHPrimitive>& bvhPrimitives, int start, int end, int depth);
    int BucketIndex(const BVHPrimitive& primitive, const Bounds3& centroidBounds, int dim) const;
    int PartitionSAH(std::vector<BVHPrimitive>& bvhPrimitives, int start, int end, const Bounds3& bounds, const Bounds3& centroidBounds, int dim) const;
    int FlattenBVH(BVHBuildNode* node, std::vector<LinearBVHNode>& nodes, int& offset) const;
};

void BuildBVH(const std::vector<Geom>& geoms, int maxPrimsInNode, std::vector<LinearBVHNode>& nodes, std::vector<int>& orderedGeomIndices, int maxDepth = 64);
