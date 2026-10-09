#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
	void loadMeshFromOBJ(const std::string& objName, Geom& geom);
	int loadTexture(const std::string& texName);
    int buildBVH(int start, int end);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
	std::vector<Texture> textures;
	std::vector<glm::vec4> texels;
    int envTexId = -1;
    float envIntensity = 1.f;
    std::vector<BVHNode> bvhNodes;
    std::vector<int> lightIds;
    RenderState state;
};
