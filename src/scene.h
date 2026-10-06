#pragma once

#include "sceneStructs.h"
#include <vector>
#include <unordered_map>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadOBJ(const std::string& path, Geom& geom);
    int buildBVH(int first, int count);
    void buildBVHNode(int nodeIdx, int first, int count, int depth);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;

    // Equirectangular environment map (empty = black background)
    std::vector<glm::vec3> envMap;
    std::vector<float> envCdf;   // size w*h+1, normalized

    std::vector<int> lights;   // indices of emissive cube/sphere geoms

    std::vector<glm::vec3> texPixels;     // all textures packed, linear RGB
    std::vector<TextureInfo> textures;
    std::unordered_map<std::string, int> texCache;
    int loadTexture(const std::string& path, bool srgb = true);

    int envWidth = 0;
    int envHeight = 0;
    float envIntensity = 1.0f;
    float envRotation = 0.0f;   // radians


    RenderState state;
};
