#pragma once

// OptiX launch parameters, included by a host .cpp and an OptiX-IR compile: plain structs and pointers only.

#include <optix.h>

#include "scene/sceneStructs.h"

// Per-instance data for the hit programs, indexed by optixGetInstanceId(): 20 bytes instead of the Geom's three mat4s.
struct InstanceRecord
{
    int materialId;
    int meshId;              // index into MeshBuffers::meshes; -1 for non-mesh instances
    float tangentSign;
    unsigned int lightMask;
    int lightGroup;
};

// Launch indices below numPaths are path rays; the rest are shadow rays that only test for a blocker.
struct OptixIntersectParams
{
    int iter;                              // seeds the ALPHA_BLEND test
    int numPaths;
    const PathSegment* paths;
    const ShadowRay* shadowRays;           // shadowRays[i - numPaths] for launch index i >= numPaths
    const int* shadowCount;                // the launch may be wider than this
    glm::vec3* image;
    ShadeableIntersection* intersections;
    int* materialIds;                      // materialId on a hit, numMaterials on a miss
    int numMaterials;
    const InstanceRecord* instances;
    MeshBuffers buffers;
    // For the any-hit alpha test; textures is null when the scene has none.
    const Material* materials;
    const cudaTextureObject_t* textures;
    OptixTraversableHandle handle;         // the IAS over the unit cube, unit sphere and mesh GASes
};
