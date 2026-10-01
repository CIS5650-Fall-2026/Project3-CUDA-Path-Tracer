#pragma once

#include "bvh.h"
#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadMesh(const std::string& filename, Geom& geom, bool useFileMaterials, float emittance);
    int loadTexture(const std::string& filename);
    int addTexture(const unsigned char* pixels, int width, int height, int channels);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<Texture> textures;
    std::vector<glm::vec3> texels;
    BVH bvh;
    RenderState state;
};
