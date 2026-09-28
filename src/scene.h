#pragma once

#include "sceneStructs.h"
#include <string>
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadMesh(const std::string& objPath, const glm::mat4& transform, int materialId);
    void buildBVH();

public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    RenderState state;
};
