#pragma once

#include "sceneStructs.h"

#include <string>
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objFileName, Geom& geom);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    std::vector<Material> materials;
    RenderState state;
};
