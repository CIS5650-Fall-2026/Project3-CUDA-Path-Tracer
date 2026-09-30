#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGltfWrapper(const std::string& path);
public:
    Scene(std::string filename);

    // for basic bounding box
    glm::vec3 meshBoundsMin;
    glm::vec3 meshBoundsMax;
    // for tree bounding boxes
    std::vector<node> nodes;

    // actual mesh
    std::vector<Geom> geoms; // old one square and circle
    std::vector<Triangle> triangles;

    // materials and textures
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<unsigned char> textureData;

    RenderState state;
};
