#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadOBJ(const std::string& objName, Geom& geom);
    void buildBVH(Geom& geom, int maxDepth);
    void buildBVHNode(int nodeIndex, int first, int count, int depth, int maxDepth);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    Fog fog;
    RenderState state;
};
