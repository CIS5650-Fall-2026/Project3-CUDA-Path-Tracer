#pragma once

#include "scene/sceneStructs.h"

#include <glm/glm.hpp>


__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + t * glm::normalize(r.direction);
}

__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// Every test returns the ray parameter t, or -1 on a miss. The unit cube
// spans -0.5 to 0.5, the unit sphere has radius 0.5, both centered at the origin.
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// Closest hit over every triangle of the mesh; the normal faces the ray and
// alpha cut-outs do not count. Device only: the alpha test reads textures.
__device__ float meshIntersectionTest(
    const Geom& geom,
    int geomIndex,
    const TriangleMesh& mesh,
    const MeshBuffers& buffers,
    const Material& material,
    const cudaTextureObject_t* textures,
    unsigned int alphaSeed,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec4& tangent,
    bool& outside);
