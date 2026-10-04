#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>
#include <unordered_map>

// 8-bit RGBA image, row 0 is the top
struct TextureData
{
    int width = 0;
    int height = 0;
    bool srgb = false;   // color textures are sRGB-encoded and get linearized on fetch; height maps are raw data
    std::vector<unsigned char> rgba;
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    int loadTexture(const std::string& path, bool srgb);
    std::unordered_map<std::string, int> textureCache;
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    std::vector<int> lightGeomIndices;
    std::vector<TextureData> textures;
    std::vector<float> hdriPixels;
    int hdriWidth = 0;
    int hdriHeight = 0;
    float hdriIntensity = 1.0f;
    float hdriRotation = 0.0f;   // fraction of a full turn around +Y
    RenderState state;
};
