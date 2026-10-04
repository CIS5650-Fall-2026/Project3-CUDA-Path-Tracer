#pragma once

#include "sceneStructs.h"
#include "gltf_loader.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<Triangle> triangles;
    AABB meshAABB;

    std::vector<BVHNode> bvhNodes;
};
