#include "bvh.h"

#include <cfloat>
#include <utility>

Bounds3::Bounds3() : pMin(FLT_MAX), pMax(-FLT_MAX) {}

bool Bounds3::IsEmpty() const {
    return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
}

void Bounds3::Expand(const glm::vec3& point) {
    pMin = glm::min(pMin, point);
    pMax = glm::max(pMax, point);
}

void Bounds3::Expand(const Bounds3& bounds) {
    if (bounds.IsEmpty()) {
        return;
    }
    if (IsEmpty()) {
        *this = bounds;
        return;
    }

    pMin = glm::min(pMin, bounds.pMin);
    pMax = glm::max(pMax, bounds.pMax);
}

float Bounds3::SurfaceArea() const {
    if (IsEmpty()) {
        return 0.0f;
    }

    glm::vec3 diagonal = pMax - pMin;
    return 2.0f * (diagonal.x * diagonal.y + diagonal.x * diagonal.z + diagonal.y * diagonal.z);
}

int Bounds3::MaxDimension() const {
    glm::vec3 diagonal = pMax - pMin;

    if (diagonal.x > diagonal.y && diagonal.x > diagonal.z) {
        return 0;
    }
    if (diagonal.y > diagonal.z) {
        return 1;
    }
    return 2;
}

BVHPrimitive::BVHPrimitive(int primitiveIndex, const Bounds3& bounds)
    : primitiveIndex(primitiveIndex), bounds(bounds) {}

glm::vec3 BVHPrimitive::Centroid() const {
    return 0.5f * bounds.pMin + 0.5f * bounds.pMax;
}

void BVHBuildNode::InitLeaf(int first, int count, const Bounds3& nodeBounds) {
    firstPrimOffset = first;
    nPrimitives = count;
    bounds = nodeBounds;
    children[0] = nullptr;
    children[1] = nullptr;
}

void BVHBuildNode::InitInterior(int axis, BVHBuildNode* child0, BVHBuildNode* child1) {
    children[0] = child0;
    children[1] = child1;
    splitAxis = axis;
    nPrimitives = 0;
    bounds = child0->bounds;
    bounds.Expand(child1->bounds);
}

BVH::BVH(const std::vector<Geom>& primitives, int maxPrimsInNode, int maxDepth)
    : primitives(primitives), maxPrimsInNode(maxPrimsInNode), maxDepth(maxDepth) {}

Bounds3 BVH::PrimitiveBounds(const Geom& primitive) const {
    Bounds3 bounds;

    if (primitive.type == TRIANGLE) {
        bounds.Expand(primitive.triangleVertices[0]);
        bounds.Expand(primitive.triangleVertices[1]);
        bounds.Expand(primitive.triangleVertices[2]);
        return bounds;
    }

    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                glm::vec3 corner(x == 0 ? -0.5f : 0.5f, y == 0 ? -0.5f : 0.5f, z == 0 ? -0.5f : 0.5f);
                bounds.Expand(glm::vec3(primitive.transform * glm::vec4(corner, 1.0f)));
            }
        }
    }

    return bounds;
}

BVHBuildNode* BVH::AllocateNode() {
    allocatedNodes.push_back(std::make_unique<BVHBuildNode>());
    return allocatedNodes.back().get();
}

BVHBuildNode* BVH::CreateLeaf(BVHBuildNode* node, const std::vector<BVHPrimitive>& bvhPrimitives, int start, int end, const Bounds3& bounds) {
    int firstPrimOffset = orderedPrimitives.size();
    for (int i = start; i < end; ++i) {
        orderedPrimitives.push_back(bvhPrimitives[i].primitiveIndex);
    }

    node->InitLeaf(firstPrimOffset, end - start, bounds);
    return node;
}

int BVH::BucketIndex(const BVHPrimitive& primitive, const Bounds3& centroidBounds, int dim) const {
    float offset = (primitive.Centroid()[dim] - centroidBounds.pMin[dim]) / (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]);
    int bucket = (int)(12.0f * offset);

    if (bucket >= 12) {
        return 11;
    }
    if (bucket < 0) {
        return 0;
    }
    return bucket;
}

int BVH::PartitionSAH(std::vector<BVHPrimitive>& bvhPrimitives, int start, int end, const Bounds3& bounds, const Bounds3& centroidBounds, int dim) const {
    const int nBuckets = 12;
    const int nSplits = nBuckets - 1;
    BVHSplitBucket buckets[nBuckets];

    for (int i = start; i < end; ++i) {
        int bucket = BucketIndex(bvhPrimitives[i], centroidBounds, dim);
        buckets[bucket].count++;
        buckets[bucket].bounds.Expand(bvhPrimitives[i].bounds);
    }

    float costs[nSplits] = {};
    Bounds3 boundsBelow;
    Bounds3 boundsAbove;
    int countBelow = 0;
    int countAbove = 0;

    for (int i = 0; i < nSplits; ++i) {
        boundsBelow.Expand(buckets[i].bounds);
        countBelow += buckets[i].count;
        costs[i] += countBelow * boundsBelow.SurfaceArea();
    }
    for (int i = nSplits; i >= 1; --i) {
        boundsAbove.Expand(buckets[i].bounds);
        countAbove += buckets[i].count;
        costs[i - 1] += countAbove * boundsAbove.SurfaceArea();
    }

    int minCostSplitBucket = 0;
    float minCost = costs[0];
    for (int i = 1; i < nSplits; ++i) {
        if (costs[i] < minCost) {
            minCost = costs[i];
            minCostSplitBucket = i;
        }
    }

    int nPrimitives = end - start;
    float leafCost = nPrimitives;
    minCost = 0.5f + minCost / bounds.SurfaceArea();
    if (nPrimitives <= maxPrimsInNode && minCost >= leafCost) {
        return -1;
    }

    int first = start;
    int last = end - 1;
    while (first <= last) {
        while (first <= last && BucketIndex(bvhPrimitives[first], centroidBounds, dim) <= minCostSplitBucket) {
            ++first;
        }
        while (first <= last && BucketIndex(bvhPrimitives[last], centroidBounds, dim) > minCostSplitBucket) {
            --last;
        }
        if (first < last) {
            std::swap(bvhPrimitives[first], bvhPrimitives[last]);
            ++first;
            --last;
        }
    }

    return first == start || first == end ? -1 : first;
}

BVHBuildNode* BVH::BuildRecursive(std::vector<BVHPrimitive>& bvhPrimitives, int start, int end, int depth) {
    Bounds3 bounds;
    BVHBuildNode* node = AllocateNode();

    for (int i = start; i < end; ++i) {
        bounds.Expand(bvhPrimitives[i].bounds);
    }

    int nPrimitives = end - start;
    if (bounds.SurfaceArea() == 0.0f || nPrimitives == 1 || depth >= maxDepth) {
        return CreateLeaf(node, bvhPrimitives, start, end, bounds);
    }

    Bounds3 centroidBounds;
    for (int i = start; i < end; ++i) {
        centroidBounds.Expand(bvhPrimitives[i].Centroid());
    }

    int dim = centroidBounds.MaxDimension();
    if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
        return CreateLeaf(node, bvhPrimitives, start, end, bounds);
    }

    int mid = PartitionSAH(bvhPrimitives, start, end, bounds, centroidBounds, dim);

    if (mid < 0) {
        return CreateLeaf(node, bvhPrimitives, start, end, bounds);
    }

    BVHBuildNode* child0 = BuildRecursive(bvhPrimitives, start, mid, depth + 1);
    BVHBuildNode* child1 = BuildRecursive(bvhPrimitives, mid, end, depth + 1);
    node->InitInterior(dim, child0, child1);
    return node;
}

int BVH::FlattenBVH(BVHBuildNode* node, std::vector<LinearBVHNode>& nodes, int& offset) const {
    int nodeOffset = offset++;
    LinearBVHNode& linearNode = nodes[nodeOffset];
    linearNode.bounds = node->bounds;

    if (node->nPrimitives > 0) {
        linearNode.primitivesOffset = node->firstPrimOffset;
        linearNode.secondChildOffset = -1;
        linearNode.nPrimitives = node->nPrimitives;
        linearNode.axis = 0;
    } else {
        linearNode.primitivesOffset = -1;
        linearNode.nPrimitives = 0;
        linearNode.axis = node->splitAxis;
        FlattenBVH(node->children[0], nodes, offset);
        linearNode.secondChildOffset = FlattenBVH(node->children[1], nodes, offset);
    }

    return nodeOffset;
}

void BVH::Build(std::vector<LinearBVHNode>& nodes, std::vector<int>& outputOrderedPrimitives) {
    nodes.clear();
    orderedPrimitives.clear();
    allocatedNodes.clear();

    if (primitives.empty()) {
        outputOrderedPrimitives.clear();
        return;
    }

    std::vector<BVHPrimitive> bvhPrimitives;
    bvhPrimitives.reserve(primitives.size());
    orderedPrimitives.reserve(primitives.size());

    for (int i = 0; i < primitives.size(); ++i) {
        bvhPrimitives.emplace_back(i, PrimitiveBounds(primitives[i]));
    }

    BVHBuildNode* root = BuildRecursive(bvhPrimitives, 0, bvhPrimitives.size(), 0);
    nodes.resize(allocatedNodes.size());

    int offset = 0;
    FlattenBVH(root, nodes, offset);
    outputOrderedPrimitives = std::move(orderedPrimitives);
}

void BuildBVH(const std::vector<Geom>& geoms, int maxPrimsInNode, std::vector<LinearBVHNode>& nodes, std::vector<int>& orderedGeomIndices, int maxDepth) {
    BVH bvh(geoms, maxPrimsInNode, maxDepth);
    bvh.Build(nodes, orderedGeomIndices);
}
