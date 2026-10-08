#pragma once

#include "sceneStructs.h"
#include "bvh.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <cfloat>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__host__ __device__ inline float aabbIntersectionTest(
    const glm::vec3& boundsMin,
    const glm::vec3& boundsMax,
    const glm::vec3& origin,
    const glm::vec3& invDirection,
    float tMax) {
    float tEnter = 0.0f;
    float tExit = tMax;
    for (int axis = 0; axis < 3; axis++) {
        const float t0 = (boundsMin[axis] - origin[axis]) * invDirection[axis];
        const float t1 = (boundsMax[axis] - origin[axis]) * invDirection[axis];
        tEnter = fmaxf(tEnter, fminf(t0, t1));
        tExit = fminf(tExit, fmaxf(t0, t1));
    }
    return tEnter <= tExit ? tEnter : -1.0f;
}

__host__ __device__ inline float triangleIntersectionTest(const Triangle& tri, const Ray& r, glm::vec2& bary) {
    const glm::vec3 e1 = tri.v1 - tri.v0;
    const glm::vec3 e2 = tri.v2 - tri.v0;
    const glm::vec3 p = glm::cross(r.direction, e2);
    float a = glm::dot(e1, p);

    if (fabsf(a) < FLT_EPSILON) {
        return -1.0f;
    }
    float f = 1.0f / a;
    const glm::vec3 s = r.origin - tri.v0;
    bary.x = f * glm::dot(s, p);
    if (bary.x < 0.0f || bary.x > 1.0f) {
        return -1.0f;
    }
    const glm::vec3 q = glm::cross(s, e1);
    bary.y = f * glm::dot(r.direction, q);
    if (bary.y < 0.0f || bary.x + bary.y > 1.0f) {
        return -1.0f;
    }
    float t = f * glm::dot(e2, q);
    if (t < 0) {
        return -1.0f;
    }
    return t;
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside
);

__host__ __device__ float meshIntersectionTestBVH(
    const Geom& mesh,
    const Triangle* triangles,
    const BVHNode* nodes,
    Ray r,
    float tMax,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside
);