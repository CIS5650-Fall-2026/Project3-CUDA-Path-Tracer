#pragma once

#include "sceneStructs.h"

#include <cuda_runtime.h>

/**
 * Applies a material's color texture and bump (height) map at a hit point.
 *
 * Meshes with UVs use them unless triplanar is selected.
 * Everything else uses triplanar in object space.
 *
 * @param worldPos, worldNormal  The exact hit point and geometric normal (facing the incoming ray).
 * @param albedo                 Output: texture color, or m.color if there is no color texture.
 * @param shadingNormal          Output: bumped normal, or worldNormal if there is no bump map.
 *                               Always on the same side of the surface as worldNormal.
 */
__device__ void textureSurface(
    const Material& m,
    const Geom& geom,
    const Triangle* triangles,
    int triId,
    glm::vec3 worldPos,
    glm::vec3 worldNormal,
    const cudaTextureObject_t* textures,
    glm::vec3& albedo,
    glm::vec3& shadingNormal);
