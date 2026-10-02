#pragma once

#include "geometry.h"
#include "sceneStructs.h"

#include <string>
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    SceneView view() const;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::vector<BvhNode> bvhNodes;
    std::vector<int> bvhPrimitiveIndices;
    std::vector<TextureDescriptor> textures;
    std::vector<glm::vec3> texels;
    std::vector<std::string> assets;
    int bvhMaxDepth = 60;
};
