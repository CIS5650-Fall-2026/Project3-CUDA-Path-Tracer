#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<BoundingVolumeHierarchyNode> bvhNodes;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    RenderState state;
};
