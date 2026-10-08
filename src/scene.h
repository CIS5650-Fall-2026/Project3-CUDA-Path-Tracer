#pragma once

#include "sceneStructs.h"
#include <vector>
#include "bvh.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& filename, Geom& mesh);

public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    RenderState state;
};
