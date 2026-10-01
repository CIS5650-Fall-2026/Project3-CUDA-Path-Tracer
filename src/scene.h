#pragma once

#include "sceneStructs.h"
#include "stb_image.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<TextureData> textures;
    std::vector<glm::vec3> texturePixels;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    RenderState state;
};
