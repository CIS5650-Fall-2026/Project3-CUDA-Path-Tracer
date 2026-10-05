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
    std::vector<Geom> lights;
    std::vector<Material> materials;
    std::vector<BVHNode> bvhNodes;
    std::vector<BVHPrimitive> bvhPrimitives;
    std::vector<BVHPrimitive> orderedPrims;
    std::vector<glm::vec3> textureImage;
    RenderState state;
};
