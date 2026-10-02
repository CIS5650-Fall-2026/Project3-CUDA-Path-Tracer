#pragma once
#include "sceneStructs.h"
#include <cfloat>
#include <cmath>

#define rayEpsilon 0.0001f
__host__ __device__ inline glm::vec3 transformPoint(const glm::mat4 &m, glm::vec3 p)
{
    return glm::vec3(m * glm::vec4(p, 1));
}
__host__ __device__ inline glm::vec3 transformVector(const glm::mat4 &m, glm::vec3 p)
{
    return glm::vec3(m * glm::vec4(p, 0));
}
__host__ __device__ inline bool boundsHit(glm::vec3 o, glm::vec3 d, glm::vec3 lo, glm::vec3 hi, float limit)
{
    float nearT = 0, farT = limit;
    for (int k = 0; k < 3; ++k)
    {
        if (fabsf(d[k]) < 1e-20f)
        {
            if (o[k] < lo[k] || o[k] > hi[k])
                return false;
        }
        else
        {
            float a = (lo[k] - o[k]) / d[k], b = (hi[k] - o[k]) / d[k];
            nearT = fmaxf(nearT, fminf(a, b));
            farT = fminf(farT, fmaxf(a, b));
            if (nearT > farT)
                return false;
        }
    }
    return farT > rayEpsilon;
}
__host__ __device__ inline bool intersectGeometry(const Geom &g, const Triangle *triangles, const Ray &ray, bool cull,
                                                  float limit, ShadeableIntersection &hit)
{
    glm::vec3 o = transformPoint(g.inverseTransform, ray.origin - g.velocity * ray.time);
    // Do not normalize: local t remains the world ray parameter under nonuniform scaling.
    glm::vec3 d = transformVector(g.inverseTransform, ray.direction), normal(0);
    float t = limit;
    int triangleId = -1;
    if (g.type == SPHERE)
    {
        float a = glm::dot(d, d), b = glm::dot(o, d), c = glm::dot(o, o) - 0.25f;
        float disc = b * b - a * c;
        if (disc < 0)
            return false;
        float root = sqrtf(disc), aT = (-b - root) / a, bT = (-b + root) / a;
        float candidate = aT > rayEpsilon ? aT : bT;
        if (candidate <= rayEpsilon || candidate >= t)
            return false;
        t = candidate;
        normal = glm::normalize(o + t * d);
    }
    else if (g.type == CUBE)
    {
        float entry = -FLT_MAX, exit = FLT_MAX;
        glm::vec3 en(0), ex(0);
        for (int k = 0; k < 3; ++k)
        {
            if (fabsf(d[k]) < 1e-20f)
            {
                if (o[k] < -0.5f || o[k] > 0.5f)
                    return false;
                continue;
            }
            float a = (-0.5f - o[k]) / d[k], b = (0.5f - o[k]) / d[k];
            glm::vec3 na(0), nb(0);
            na[k] = -1;
            nb[k] = 1;
            if (a > b)
            {
                float s = a;
                a = b;
                b = s;
                glm::vec3 ns = na;
                na = nb;
                nb = ns;
            }
            if (a > entry)
            {
                entry = a;
                en = na;
            }
            if (b < exit)
            {
                exit = b;
                ex = nb;
            }
            if (entry > exit)
                return false;
        }
        float candidate = entry > rayEpsilon ? entry : exit;
        if (candidate <= rayEpsilon || candidate >= t)
            return false;
        t = candidate;
        normal = entry > rayEpsilon ? en : ex;
    }
    else
    {
        if (cull && !boundsHit(o, d, g.boundsMin - glm::vec3(1e-6f), g.boundsMax + glm::vec3(1e-6f), t))
            return false;
        for (int i = g.triangleStart; i < g.triangleStart + g.triangleCount; ++i)
        {
            const Triangle &tri = triangles[i];
            glm::vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
            glm::vec3 p = glm::cross(d, e2);
            float det = glm::dot(e1, p);
            if (fabsf(det) < 1e-12f)
                continue;
            float inv = 1 / det;
            glm::vec3 s = o - tri.a;
            float u = glm::dot(s, p) * inv;
            if (u < 0 || u > 1)
                continue;
            glm::vec3 q = glm::cross(s, e1);
            float v = glm::dot(d, q) * inv;
            if (v < 0 || u + v > 1)
                continue;
            float candidate = glm::dot(e2, q) * inv;
            if (candidate > rayEpsilon && candidate < t)
            {
                t = candidate;
                normal = glm::normalize(glm::cross(e1, e2));
                triangleId = i;
            }
        }
        if (triangleId < 0)
            return false;
    }
    hit.t = t;
    hit.point = ray.origin + t * ray.direction;
    hit.materialId = g.materialid;
    hit.triangleId = triangleId;
    hit.outwardNormal = glm::normalize(transformVector(g.invTranspose, normal));
    hit.frontFace = glm::dot(ray.direction, hit.outwardNormal) < 0;
    hit.surfaceNormal = hit.frontFace ? hit.outwardNormal : -hit.outwardNormal;
    return true;
}
// Intersection overloads return exact surface points and entry/exit metadata
// while accounting for object motion.
__host__ __device__ inline float boxIntersectionTest(Geom box, Ray ray,
    glm::vec3& point, glm::vec3& normal, bool& outside, ShadeableIntersection& hit)
{
    if (!intersectGeometry(box, nullptr, ray, false, FLT_MAX, hit)) return -1;
    point = hit.point;
    normal = hit.surfaceNormal;
    outside = hit.frontFace;
    return hit.t;
}
__host__ __device__ inline float sphereIntersectionTest(Geom sphere, Ray ray,
    glm::vec3& point, glm::vec3& normal, bool& outside, ShadeableIntersection& hit)
{
    if (!intersectGeometry(sphere, nullptr, ray, false, FLT_MAX, hit)) return -1;
    point = hit.point;
    normal = hit.surfaceNormal;
    outside = hit.frontFace;
    return hit.t;
}
__host__ __device__ inline ShadeableIntersection traceScene(const Ray &ray, const Geom *geoms, int count,
                                                            const Triangle *triangles, bool cull, float limit = FLT_MAX)
{
    ShadeableIntersection nearest{};
    nearest.t = -1;
    nearest.materialId = -1;
    for (int i = 0; i < count; ++i)
    {
        ShadeableIntersection h{};
        if (intersectGeometry(geoms[i], triangles, ray, cull, limit, h))
        {
            h.geomId = i;
            nearest = h;
            limit = h.t;
        }
    }
    return nearest;
}
__host__ __device__ inline float triangleArea(const Geom &g, const Triangle &t)
{
    return 0.5f *
           glm::length(glm::cross(transformVector(g.transform, t.b - t.a), transformVector(g.transform, t.c - t.a)));
}
__host__ __device__ inline float cubeArea(const Geom &g)
{
    return 2 * (g.scale.x * g.scale.y + g.scale.x * g.scale.z + g.scale.y * g.scale.z);
}
__host__ __device__ inline float lightAreaPdf(const Geom &g, const Triangle *triangles, glm::vec3 point, float time,
                                              int triangleId)
{
    if (g.type == CUBE)
        return 1 / cubeArea(g);
    if (g.type == MESH)
        return 1 / (g.triangleCount * triangleArea(g, triangles[triangleId]));
    glm::vec3 localNormal = glm::normalize(transformPoint(g.inverseTransform, point - g.velocity * time));
    float jacobian = g.scale.x * g.scale.y * g.scale.z * glm::length(transformVector(g.invTranspose, localNormal));
    return 1 / (3.14159265359f * jacobian);
}
struct LightSample
{
    glm::vec3 point, normal;
    float areaPdf;
};
__host__ __device__ inline LightSample sampleLight(const Geom &g, const Triangle *triangles, float time, float choice,
                                                   float u, float v)
{
    glm::vec3 p(0), n(0);
    int triId = -1;
    if (g.type == SPHERE)
    {
        float z = 1 - 2 * u, r = sqrtf(fmaxf(0, 1 - z * z)), phi = 6.28318530718f * v;
        n = glm::vec3(r * cosf(phi), r * sinf(phi), z);
        p = 0.5f * n;
    }
    else if (g.type == CUBE)
    {
        float weights[3] = {g.scale.y * g.scale.z, g.scale.x * g.scale.z, g.scale.x * g.scale.y};
        float selected = choice * cubeArea(g);
        int face = 5;
        for (int i = 0; i < 6; ++i)
        {
            if (selected < weights[i / 2])
            {
                face = i;
                break;
            }
            selected -= weights[i / 2];
        }
        int axis = face / 2;
        n[axis] = (face % 2) ? 1.0f : -1.0f;
        p[axis] = 0.5f * n[axis];
        p[(axis + 1) % 3] = u - 0.5f;
        p[(axis + 2) % 3] = v - 0.5f;
    }
    else
    {
        triId = g.triangleStart + int(choice * g.triangleCount);
        const auto &t = triangles[triId];
        float root = sqrtf(u);
        p = (1 - root) * t.a + root * (1 - v) * t.b + root * v * t.c;
        n = glm::normalize(glm::cross(t.b - t.a, t.c - t.a));
    }
    LightSample s;
    s.point = transformPoint(g.transform, p) + g.velocity * time;
    s.normal = glm::normalize(transformVector(g.invTranspose, n));
    s.areaPdf = lightAreaPdf(g, triangles, s.point, time, triId);
    return s;
}
