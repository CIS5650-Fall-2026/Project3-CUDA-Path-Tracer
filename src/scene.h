#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    std::string vdbFileName;
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Volume> volumes;
    std::vector<Material> materials;
    std::vector<Light> lights;
    RenderState state;

    std::string vdbFileNameCurrent;

    void setFrame(int i);
};
