#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene {
private:
  void loadFromJSON(const std::string &jsonName);
  void loadGLTF(const std::string &path, bool normalize,
                const std::string &nodeName, Geom &geom);

public:
  Scene(std::string filename);

  std::vector<Geom> geoms;
  std::vector<Triangle> triangles;
  std::vector<OctNode> octreeNodes;
  std::vector<int> octreeTriIndices; // indices into triangles grouped per leaf
  std::vector<Material> materials;
  RenderState state;
};
