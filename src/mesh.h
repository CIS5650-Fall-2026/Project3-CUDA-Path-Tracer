#pragma once

#include "intersections.h"
#include <cfloat>

// The ray direction stays unnormalized in object space, preserving world-space t.
__device__ inline bool meshBoundsHit(const MeshBVHNode& node, const Ray& ray, float closest)
{
    float nearT = 0.00001f;
    float farT = closest;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (fabsf(ray.direction[axis]) < 1e-9f)
        {
            if (ray.origin[axis] < node.minimum[axis] || ray.origin[axis] > node.maximum[axis])
                return false;
        }
        else
        {
            float a = (node.minimum[axis] - ray.origin[axis]) / ray.direction[axis];
            float b = (node.maximum[axis] - ray.origin[axis]) / ray.direction[axis];
            nearT = fmaxf(nearT, fminf(a, b));
            farT = fminf(farT, fmaxf(a, b));
            if (farT < nearT) return false;
        }
    }
    return true;
}

__device__ inline bool meshTriangleHit(const Triangle& triangle, const Ray& ray,
    float& closest, glm::vec3& smoothNormal, glm::vec3& geometricNormal)
{
    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;
    glm::vec3 p = glm::cross(ray.direction, edge2);
    float determinant = glm::dot(edge1, p);
    if (fabsf(determinant) < 1e-10f) return false;
    float inverse = 1.0f / determinant;
    glm::vec3 offset = ray.origin - triangle.v0;
    float u = glm::dot(offset, p) * inverse;
    if (u < 0.0f || u > 1.0f) return false;
    glm::vec3 q = glm::cross(offset, edge1);
    float v = glm::dot(ray.direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = glm::dot(edge2, q) * inverse;
    if (t <= 0.00001f || t >= closest) return false;
    closest = t;
    geometricNormal = glm::normalize(glm::cross(edge1, edge2));
    smoothNormal = glm::normalize((1.0f - u - v) * triangle.n0 + u * triangle.n1 + v * triangle.n2);
    if (glm::dot(smoothNormal, geometricNormal) < 0.0f) smoothNormal = -smoothNormal;
    return true;
}

__device__ inline float meshIntersectionTest(const Geom& geom, const Ray& ray,
    const Triangle* triangles, const MeshBVHNode* nodes, bool useBVH,
    glm::vec3& point, glm::vec3& normal, bool& outside, float maxDistance = FLT_MAX)
{
    Ray local;
    local.origin = multiplyMV(geom.inverseTransform, glm::vec4(ray.origin, 1.0f));
    local.direction = multiplyMV(geom.inverseTransform, glm::vec4(ray.direction, 0.0f));
    float closest = maxDistance;
    glm::vec3 smoothNormal, geometricNormal;
    bool hit = false;
    if (useBVH)
    {
        int cursor = geom.bvhRoot;
        int end = nodes[cursor].escape;
        while (cursor < end)
        {
            const MeshBVHNode& node = nodes[cursor];
            if (!meshBoundsHit(node, local, closest))
            {
                cursor = node.escape;
                continue;
            }
            for (int i = 0; i < node.count; ++i)
                hit = meshTriangleHit(triangles[node.first + i], local, closest,
                    smoothNormal, geometricNormal) || hit;
            ++cursor;
        }
    }
    else
    {
        for (int i = 0; i < geom.triangleCount; ++i)
            hit = meshTriangleHit(triangles[geom.triangleStart + i], local, closest,
                smoothNormal, geometricNormal) || hit;
    }
    if (!hit) return -1.0f;
    point = ray.origin + closest * ray.direction;
    glm::vec3 worldGeometric = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(geometricNormal, 0.0f)));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(smoothNormal, 0.0f)));
    outside = glm::dot(ray.direction, worldGeometric) < 0.0f;
    if (glm::dot(normal, ray.direction) > 0.0f) normal = -normal;
    return closest;
}
