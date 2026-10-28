#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& path, Geom& geom);
    int loadTexture(const std::string& path);
    int buildBVH(int triStart, int triCount, int depth);

public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;        // every mesh's hierarchy, back to back
    int bvhLeafSize = 4;                  // a node with this many triangles or fewer becomes a leaf
    int bvhMaxDepth = 24;                 // splitting stops at this depth, the root being depth 0
    std::vector<glm::vec3> texels;        // every image's pixels, back to back
    std::vector<TextureInfo> textures;    // where each image sits inside texels
    std::vector<int> lights;              // indices into geoms of every emissive box
};
