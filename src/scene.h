#pragma once

#include "sceneStructs.h"

#include "tinygltf/tiny_gltf_v3.h";

#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
	bool loadModelGLTF(const std::string& filename, tg3_model& model);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;

	std::vector<Triangle> triangles;
	std::vector<BVHNode> bvhNodes;
	int bvhRootNodeIndex;

    RenderState state;
};
