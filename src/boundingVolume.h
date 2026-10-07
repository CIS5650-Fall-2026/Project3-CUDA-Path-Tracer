#pragma once

#include "sceneStructs.h"

#include <vector>

int buildBoundingVolumeHierarchy(std::vector<Triangle>& triangles, int start, int count, std::vector<BoundingVolumeHierarchyNode>& nodes);
