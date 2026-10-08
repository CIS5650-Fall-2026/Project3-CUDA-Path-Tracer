#pragma once

#include "sceneStructs.h"

void sortIntersections(ShadeableIntersection* start, ShadeableIntersection* end);
PathSegment* removeUnusedPaths(PathSegment* start, PathSegment* end);