#include "intersections.h"
#include <cfloat>
#include <cmath>

__host__ __device__ float boxIntersectionTest(
    Geom box, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal,
    bool& outside)
{
    intersectionPoint = glm::vec3(0.0f);
    normal = glm::vec3(0.0f);
    outside = true;
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(
        multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -FLT_MAX;
    float tmax = FLT_MAX;
    glm::vec3 entryNormal(0.0f), exitNormal(0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        if (q.direction[axis] == 0.0f) {
            // A parallel ray either stays in this slab or never reaches it.
            if (q.origin[axis] < -0.5f || q.origin[axis] > 0.5f) return -1.0f;
            continue;
        }
        const float t1 = (-0.5f - q.origin[axis]) / q.direction[axis];
        const float t2 = ( 0.5f - q.origin[axis]) / q.direction[axis];
        const float nearT = fminf(t1, t2);
        const float farT = fmaxf(t1, t2);
        glm::vec3 nearNormal(0.0f);
        nearNormal[axis] = t1 < t2 ? -1.0f : 1.0f;
        // Keep the full slab interval, including negative entry distances.
        // After all slabs, use the entry outside the box or the exit inside.
        if (nearT > tmin) {
            tmin = nearT;
            entryNormal = nearNormal;
        }
        if (farT < tmax) {
            tmax = farT;
            exitNormal = -nearNormal;
        }
        if (tmin > tmax) return -1.0f;
    }
    if (tmax <= 0.0f) return -1.0f;
    outside = tmin > 0.0f;
    const float t = outside ? tmin : tmax;
    const glm::vec3 objectNormal = outside ? entryNormal : exitNormal;
    intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
    normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(objectNormal, 0.0f)));
    return glm::length(intersectionPoint - r.origin);
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal,
    bool& outside)
{
    intersectionPoint = glm::vec3(0.0f);
    normal = glm::vec3(0.0f);
    outside = true;
    Ray q;
    q.origin = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(
        multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    const float b = glm::dot(q.origin, q.direction);
    const float radicand = b * b - (glm::dot(q.origin, q.origin) - 0.25f);
    if (radicand < 0.0f) return -1.0f;
    const float root = sqrtf(radicand);
    const float nearT = -b - root;
    const float farT = -b + root;
    if (farT <= 0.0f) return -1.0f;
    outside = nearT > 0.0f;
    const glm::vec3 objectPoint = getPointOnRay(q, outside ? nearT : farT);
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objectPoint, 1.0f));
    // Always outward, including rays that start inside the sphere.
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objectPoint, 0.0f)));
    return glm::length(intersectionPoint - r.origin);
}
