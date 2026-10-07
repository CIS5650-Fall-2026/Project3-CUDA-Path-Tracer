#include "boundingVolume.h"
#include <algorithm>
#include <cfloat>

int buildBoundingVolumeHierarchy(std::vector<Triangle>& triangles, int start, int count, std::vector<BoundingVolumeHierarchyNode>& nodes) {
    glm::vec3 boxMin(FLT_MAX);
    glm::vec3 boxMax(-FLT_MAX);
    glm::vec3 centerMin(FLT_MAX);
    glm::vec3 centerMax(-FLT_MAX);
    for (int i = start; i < start + count; i++) {
        Triangle tri = triangles[i];
        glm::vec3 triMin = glm::min(tri.v0, glm::min(tri.v1, tri.v2));
        glm::vec3 triMax = glm::max(tri.v0, glm::max(tri.v1, tri.v2));
        boxMin = glm::min(boxMin, triMin);
        boxMax = glm::max(boxMax, triMax);
        glm::vec3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
        centerMin = glm::min(centerMin, centroid);
        centerMax = glm::max(centerMax, centroid);
    }
    if (count <= 4) {
        BoundingVolumeHierarchyNode leafNode;
        leafNode.bboxMin = boxMin;
        leafNode.bboxMax = boxMax;
        leafNode.left = -1;
        leafNode.right = -1;
        leafNode.start = start;
        leafNode.triCount = count;

        nodes.push_back(leafNode);
        return (int)nodes.size() - 1;
    }
    BoundingVolumeHierarchyNode node;
    node.bboxMin = boxMin;
    node.bboxMax = boxMax;
    node.start = 0;
    node.triCount = 0;
    int index = (int)nodes.size();
    nodes.push_back(node);
    glm::vec3 spread = centerMax - centerMin;
    int axis = 0;
    if (spread.y > spread.x) axis = 1;
    if (spread.z > spread[axis]) axis = 2;
    std::sort(triangles.begin() + start, triangles.begin() + start + count,
        [axis](const Triangle& a, const Triangle& b) {
            glm::vec3 centerA = (a.v0 + a.v1 + a.v2) / 3.0f;
            glm::vec3 centerB = (b.v0 + b.v1 + b.v2) / 3.0f;
            return centerA[axis] < centerB[axis];
        });
//recursive step
    int mid = start + count / 2;
    int leftChild = buildBoundingVolumeHierarchy(triangles, start, mid - start, nodes);
    int rightChild = buildBoundingVolumeHierarchy(triangles, mid, start + count - mid, nodes);
    nodes[index].left = leftChild;
    nodes[index].right = rightChild;
    return index;
}