#pragma once

#include "sceneStructs.h"
#include <vector>

struct EnvironmentMap
{
    std::vector<glm::vec3> pixels;
    int width = 0;
    int height = 0;
    float intensity = 1.0f;
    float rotation = 0.0f;
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadObj(const std::string& filename, const glm::mat4& objectTransform, int materialId);
    void buildBvh(int leafSize);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<Primitive> primitives;
    std::vector<BvhNode> bvhNodes;
    EnvironmentMap environment;
    bool enableCompaction = true;
    bool sortByMaterial = true;
    bool useBvh = true;
    bool enableDepthOfField = false;
    bool enableEnvironment = false;
    float bvhBuildMs = 0.0f;
    int bvhLeafCount = 0;
    int bvhMaxDepth = 0;
    RenderState state;
};
