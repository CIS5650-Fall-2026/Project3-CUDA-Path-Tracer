#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    // parsed JSON captured at load time, excluding target samples,
    // output filename, and material-sort toggle
    std::string checkpointSceneIdentity;
};
