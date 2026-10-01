#include "bvh.h"

#include <algorithm>
#include <cfloat>

static glm::vec3 centroid(const Triangle& triangle)
{
    return (triangle.positions[0] + triangle.positions[1] + triangle.positions[2]) / 3.0f;
}

int BVH::build(std::vector<Triangle>& triangles, int start, int end, int depth)
{
    BVHNode node;
    node.boundsMin = glm::vec3(FLT_MAX);
    node.boundsMax = glm::vec3(-FLT_MAX);
    glm::vec3 centroidMin(FLT_MAX);
    glm::vec3 centroidMax(-FLT_MAX);
    for (int i = start; i < end; i++)
    {
        for (const glm::vec3& position : triangles[i].positions)
        {
            node.boundsMin = glm::min(node.boundsMin, position);
            node.boundsMax = glm::max(node.boundsMax, position);
        }
        centroidMin = glm::min(centroidMin, centroid(triangles[i]));
        centroidMax = glm::max(centroidMax, centroid(triangles[i]));
    }
    node.left = -1;
    node.right = -1;
    node.triangleStart = start;
    node.triangleCount = end - start;

    int index = nodes.size();
    nodes.push_back(node);
    if (end - start <= leafSize || depth >= maxDepth)
    {
        return index;
    }

    glm::vec3 extent = centroidMax - centroidMin;
    int axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);
    int mid = (start + end) / 2;
    std::nth_element(triangles.begin() + start, triangles.begin() + mid, triangles.begin() + end,
        [axis](const Triangle& a, const Triangle& b) { return centroid(a)[axis] < centroid(b)[axis]; });

    int left = build(triangles, start, mid, depth + 1);
    int right = build(triangles, mid, end, depth + 1);
    nodes[index].left = left;
    nodes[index].right = right;
    nodes[index].triangleCount = 0;
    return index;
}

void BVH::initDevice()
{
    cudaMalloc(&dev_nodes, nodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
}

void BVH::freeDevice()
{
    cudaFree(dev_nodes);
    dev_nodes = NULL;
}
