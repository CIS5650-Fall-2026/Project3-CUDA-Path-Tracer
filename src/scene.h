#pragma once

#include "sceneStructs.h"
#include "tiny_gltf_v3.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadMaterials(const tg3_model* model);
    void traverseNode(tg3_model* model, uint32_t nodeIndex, glm::mat4 parentTransform, int jsonMaterialId, int materialOffset);
    void traverseMesh(tg3_model* model, uint32_t meshIndex, glm::mat4 worldTransform, int jsonMaterialId, int materialOffset);
    void loadGLTFMesh(const std::string& gltfName, int jsonMaterialId, glm::mat4 baseTransform);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<LinearBVHNode> bvhNodes;
    RenderState state;
};
