#pragma once

#include "scene/sceneStructs.h"

// Path buffers and the device-wide operations on them. CUB stays in this
// translation unit: one algorithm instantiation costs ~10 s of nvcc.
// Shading and the sort write into spare buffers that then swap with the
// caller's pointers; the spares and CUB storage are shared, so every call
// goes to the same stream.

constexpr int PATH_BLOCK_SIZE = 128;

// Workspace for up to maxPaths paths: spare path and intersection buffers,
// sort arrays, survivor counters, CUB storage.
void wavefrontInit(int maxPaths);
void wavefrontFree();

// Where shadeMaterial appends the paths that survive the bounce. Three
// counters rotate by bounce so the one being zeroed is never one a queued
// kernel still reads.
struct SurvivorBuffer
{
    PathSegment* paths;
    int* count;
    int* nextCount;
};
SurvivorBuffer wavefrontSurvivors();
// The same counters with paths itself as the buffer (--no-compact).
SurvivorBuffer wavefrontSurvivorsInPlace(PathSegment* paths);

// Shadow rays shadeMaterial queues for the next OptiX launch, one per path at most.
struct ShadowQueue
{
    ShadowRay* rays;
    int* count;
    int* nextCount;
};
ShadowQueue wavefrontShadowQueue();
// After the shade launch: returns the queue it filled and moves the counters on.
ShadowQueue wavefrontSwapShadowQueue();

// After the shade launch: swaps the survivors into paths and returns their
// count (reading it synchronizes).
int wavefrontSwapPaths(PathSegment*& paths, cudaStream_t stream = 0);
// The survivor count after a shade launch into wavefrontSurvivorsInPlace, no swap.
int wavefrontCountSurvivors(cudaStream_t stream = 0);

// Sorts paths and intersections by material key (materialId on hit,
// numMaterials on miss) over the low keyBits bits. materialIds stays unsorted.
void sortMaterials(int numPaths, int keyBits, const int* materialIds,
                   ShadeableIntersection*& intersections, PathSegment*& paths,
                   cudaStream_t stream = 0);
