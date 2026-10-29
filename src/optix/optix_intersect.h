#pragma once

#include "scene/sceneStructs.h"

#include <cuda_runtime.h>

class Scene;

// OptiX intersection stage, same contract as computeIntersections: t = -1 and materialIds = numMaterials on a miss.
// The host side is a .cpp, not a .cu: optix.h switches to the device API whenever __CUDACC__ is defined.

// Builds the context, acceleration structures, pipeline and SBT; false, after printing why, when any step fails.
// buffers, materials and textures must outlive this stage; textures may be null.
bool optixIntersectInit(const Scene* scene, const MeshBuffers& buffers, const Material* materials,
                        const cudaTextureObject_t* textures, bool validation);
void optixIntersectFree();

// Traces the path rays and the *shadowCount shadow rays in one launch; an unblocked shadow ray adds to image.
// shadowBound is a host-side upper bound on *shadowCount that sets the launch width; 0 traces no shadow rays.
void optixIntersect(int iter, int numPaths, const PathSegment* paths,
                    ShadeableIntersection* intersections, int* materialIds,
                    int numMaterials, const ShadowRay* shadowRays, const int* shadowCount,
                    int shadowBound, glm::vec3* image, cudaStream_t stream = 0);
