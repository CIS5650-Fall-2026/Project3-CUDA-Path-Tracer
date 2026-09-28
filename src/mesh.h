#pragma once

#include "sceneStructs.h"

// Append world-space triangles and fill in this object's bounds and range.
void loadGltf(const std::string& filename, Geom& geom, std::vector<Triangle>& triangles);
