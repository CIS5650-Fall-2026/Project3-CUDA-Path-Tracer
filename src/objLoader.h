#pragma once
#include "sceneStructs.h"
#include <string>
#include <vector>

void loadObj(
  const std::string& filename,
  const glm::mat4& objectTransform,
  std::vector<Triangle>& triangles
);
