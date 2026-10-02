#pragma once

#include "sceneStructs.h"

#include <cfloat>
#include <vector>

struct TextureDescriptor
{
    int width = 0;
    int height = 0;
    int offset = 0;
};

struct BvhNode
{
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;
};

struct Hit
{
    float t = FLT_MAX;
    int primitiveId = -1;
    int materialId = -1;
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 geometricNormal = glm::vec3(0.0f);
    glm::vec3 shadingNormal = glm::vec3(0.0f);
    glm::vec3 dpdu = glm::vec3(0.0f);
    glm::vec3 dpdv = glm::vec3(0.0f);
    glm::vec2 uv = glm::vec2(0.0f);
    bool frontFace = false;
};

struct SceneView
{
    const Geom* geoms = nullptr;
    int geomCount = 0;
    const Material* materials = nullptr;
    int materialCount = 0;
    const BvhNode* nodes = nullptr;
    int nodeCount = 0;
    const int* primitiveIndices = nullptr;
    const TextureDescriptor* textures = nullptr;
    int textureCount = 0;
    const glm::vec3* texels = nullptr;
    int texelCount = 0;
};

// Builds a binary BVH whose root, when present, is node zero. Leaf nodes have
// count > 0 and left == right == -1. Internal nodes have count == 0.
void buildBvh(const std::vector<Geom>& geoms,
              std::vector<BvhNode>& nodes,
              std::vector<int>& primitiveIndices,
              int maxDepth = 60);
