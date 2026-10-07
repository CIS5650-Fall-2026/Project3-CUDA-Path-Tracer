#include "bvh.h"

#include "scene.h"

BVH::BVH()
    : rootNodeIdx(0), nodesUsed(0), nodes()
{
}

void BVH::UpdateNodeBounds(Scene *scene, uint32_t nodeIdx)
{
    BVHNode &node = nodes[nodeIdx];
    node.aabbMin = glm::vec3(1e30f);
    node.aabbMax = glm::vec3(-1e30f);

    for (uint32_t i = 0; i < node.triCount; ++i)
    {
        const Triangle &leafTri = scene->triangles[node.firstTriIdx + i];
        node.aabbMin = glm::min(node.aabbMin, leafTri.v0);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v1);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v2);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v0);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v1);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v2);
    }
}

void BVH::Subdivide(Scene *scene, uint32_t nodeIdx)
{
    BVHNode &node = nodes[nodeIdx];
    if (node.triCount <= 2)
    {
        return;
    }

    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x)
    {
        axis = 1;
    }
    if (extent.z > extent[axis])
    {
        axis = 2;
    }

    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

    int i = (int)node.firstTriIdx;
    int j = i + (int)node.triCount - 1;
    while (i <= j)
    {
        if (scene->triangles[i].centroid[axis] < splitPos)
        {
            ++i;
        }
        else
        {
            std::swap(scene->triangles[i], scene->triangles[j--]);
        }
    }

    uint32_t leftCount = (uint32_t)(i - (int)node.firstTriIdx);
    if (leftCount == 0 || leftCount == node.triCount)
    {
        return;
    }

    uint32_t leftChildIdx = (uint32_t)nodesUsed++;
    uint32_t rightChildIdx = (uint32_t)nodesUsed++;

    nodes[leftChildIdx].firstTriIdx = node.firstTriIdx;
    nodes[leftChildIdx].triCount = leftCount;
    nodes[leftChildIdx].leftNode = 0;

    nodes[rightChildIdx].firstTriIdx = (uint32_t)i;
    nodes[rightChildIdx].triCount = node.triCount - leftCount;
    nodes[rightChildIdx].leftNode = 0;

    node.leftNode = leftChildIdx;
    node.triCount = 0;

    UpdateNodeBounds(scene, leftChildIdx);
    UpdateNodeBounds(scene, rightChildIdx);

    Subdivide(scene, leftChildIdx);
    Subdivide(scene, rightChildIdx);
}

void BVH::BuildBVH(Scene *scene)
{
    nodes.clear();
    nodesUsed = 0;

    if (scene->triangles.empty())
    {
        return;
    }

    for (Triangle &triangle : scene->triangles)
    {
        triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
    }

    nodes.resize(scene->triangles.size() * 2 - 1);
    nodesUsed = 1;

    BVHNode &root = nodes[rootNodeIdx];
    root.leftNode = 0;
    root.firstTriIdx = 0;
    root.triCount = (uint32_t)scene->triangles.size();

    UpdateNodeBounds(scene, rootNodeIdx);
    Subdivide(scene, rootNodeIdx);

    nodes.resize(nodesUsed);
}
