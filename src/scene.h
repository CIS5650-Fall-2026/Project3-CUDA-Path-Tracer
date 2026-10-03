#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& filename, const glm::mat4& transform, bool receiveAccentLights, float atmosphereStrength);
    void buildBVH();
    int buildBVHNode(int first, int count);
    int loadTexture(const std::string& filename, bool isColor);
    void computeSceneRadius();
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    RenderState state;

    std::vector<BVHNode> bvhNodes;
    std::vector<PrimitiveRef> bvhPrimitives;

    std::vector<Texture> textures;
    std::vector<glm::vec3> texturePixels;

    std::vector<DistantLight> distantLights;
    float sceneRadius = 0.0f;
};
