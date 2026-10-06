#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
// for the remove_if stream compaction call
struct IsDead {
    __host__ __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces <= 0;
    }
};
// for thrust sorting comparison
struct MaterialCmp {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};