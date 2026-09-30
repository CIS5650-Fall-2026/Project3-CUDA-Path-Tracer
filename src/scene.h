#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadMesh(const std::string& objPath, bool smoothNormals, Geom& mesh);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;    // all mesh triangles, meshes index into this
    RenderState state;
};
