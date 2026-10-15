#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
public:
    Scene(std::string filename);

    std::vector<MeshData> meshes;
    std::vector<MeshInstance> meshInstances;

    // NCHORTEK TODO: I dont think we need geoms anymore
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

private:
    static constexpr uint32_t kCubeMeshIndex = 0;
    static constexpr uint32_t kSphereMeshIndex = 1;
    static constexpr uint32_t kIcosphereSubdivisions = 4;

    void loadFromJSON(const std::string& jsonName);
};
