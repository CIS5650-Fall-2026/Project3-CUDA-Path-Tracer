#pragma once

#include <cstdint>
#include <vector>

#include "glm/glm.hpp"

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

class Scene;

struct BVHNode
{
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    uint32_t leftNode, firstTriIdx, triCount;
    CUDA_HOST_DEVICE bool isLeaf() const { return triCount > 0; }
};

class BVH {
private:
    int rootNodeIdx;
    int nodesUsed;

    void UpdateNodeBounds(Scene *scene, uint32_t nodeIdx);
    void Subdivide(Scene *scene, uint32_t nodeIdx);
public:
    BVH();
    void BuildBVH(Scene *scene);

    std::vector<BVHNode> nodes;
};
