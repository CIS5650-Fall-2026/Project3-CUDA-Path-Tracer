#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <cfloat>

__host__ __device__ inline bool intersectAABB(
    const Ray &ray,
    const glm::vec3 &boxMin,
    const glm::vec3 &boxMax,
    float &tminOut)
{
    float tnear = -FLT_MAX;
    float tfar = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float origin = ray.origin[axis];
        const float direction = ray.direction[axis];
        if (fabsf(direction) < 1e-8f)
        {
            if (origin < boxMin[axis] || origin > boxMax[axis])
                return false;
            continue;
        }

        const float t0 = (boxMin[axis] - origin) / direction;
        const float t1 = (boxMax[axis] - origin) / direction;
        tnear = glm::max(tnear, glm::min(t0, t1));
        tfar = glm::min(tfar, glm::max(t0, t1));
        if (tnear > tfar)
            return false;
    }

    tminOut = tnear;
    return tfar >= glm::max(0.0f, tnear);
}

__host__ __device__ inline bool intersectTriangle(
    const Ray &ray,
    const Triangle &triangle,
    float &t,
    glm::vec3 &normal,
    glm::vec2 &uv)
{
    const glm::vec3 edge1 = triangle.p1 - triangle.p0;
    const glm::vec3 edge2 = triangle.p2 - triangle.p0;
    const glm::vec3 pvec = glm::cross(ray.direction, edge2);
    const float determinant = glm::dot(edge1, pvec);
    if (fabsf(determinant) < 1e-8f)
        return false;

    const float inverseDeterminant = 1.0f / determinant;
    const glm::vec3 tvec = ray.origin - triangle.p0;
    const float u = glm::dot(tvec, pvec) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f)
        return false;

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(ray.direction, qvec) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    const float hitT = glm::dot(edge2, qvec) * inverseDeterminant;
    if (hitT <= 0.001f)
        return false;

    const float w = 1.0f - u - v;
    t = hitT;
    normal = glm::normalize(w * triangle.n0 + u * triangle.n1 + v * triangle.n2);
    uv = w * triangle.uv0 + u * triangle.uv1 + v * triangle.uv2;
    return true;
}

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
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside);

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
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside);


__host__ __device__ float meshIntersectionTest(
    Geom mesh, Ray r, const Triangle* tris, const BVHNode* nodes, int accelMode,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside);