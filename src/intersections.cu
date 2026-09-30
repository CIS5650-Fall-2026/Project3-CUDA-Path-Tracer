#include "intersections.h"

#include <cfloat>

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool aabbIntersectionTest(
    glm::vec3 boxMin,
    glm::vec3 boxMax,
    Ray r)
{
    // Slab test: on each axis, when does the ray enter/leave the box?
    // It's a hit if the last "enter" comes before the first "leave"
    glm::vec3 invDir = 1.0f / r.direction;
    glm::vec3 t0 = (boxMin - r.origin) * invDir;
    glm::vec3 t1 = (boxMax - r.origin) * invDir;
    glm::vec3 tSmall = glm::min(t0, t1);
    glm::vec3 tBig = glm::max(t0, t1);

    float tEnter = glm::max(tSmall.x, glm::max(tSmall.y, tSmall.z));
    float tExit = glm::min(tBig.x, glm::min(tBig.y, tBig.z));

    // tExit < 0 means the box is behind us
    return tExit >= glm::max(tEnter, 0.0f);
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    bool bboxCulling)
{
    // Miss the box = miss every triangle inside, skip them all
    if (bboxCulling && !aabbIntersectionTest(mesh.bboxMin, mesh.bboxMax, r))
    {
        return -1.0f;
    }

    float tClosest = FLT_MAX;
    int hitIndex = -1;
    float hitU = 0.0f;  // how much of v1 (barycentric)
    float hitV = 0.0f;  // how much of v2 (barycentric)

    for (int i = mesh.triStart; i < mesh.triStart + mesh.triCount; ++i)
    {
        const Triangle& tri = triangles[i];

        // glm only hits front faces, so if that misses, swap two corners
        // and try the back side. result = (bary x, bary y, distance)
        glm::vec3 result;
        float u, v;
        if (glm::intersectRayTriangle(r.origin, r.direction, tri.v0, tri.v1, tri.v2, result))
        {
            u = result.x;
            v = result.y;
        }
        else if (glm::intersectRayTriangle(r.origin, r.direction, tri.v0, tri.v2, tri.v1, result))
        {
            u = result.y;   // corners were swapped, swap back
            v = result.x;
        }
        else
        {
            continue;
        }

        float t = result.z;
        if (t > 0.0001f && t < tClosest)
        {
            tClosest = t;
            hitIndex = i;
            hitU = u;
            hitV = v;
        }
    }

    if (hitIndex == -1)
    {
        return -1.0f;
    }

    const Triangle& tri = triangles[hitIndex];
    intersectionPoint = r.origin + tClosest * r.direction;

    // Inside or outside? Ask the real face normal (winding order = which way is out)
    glm::vec3 faceNormal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    outside = glm::dot(r.direction, faceNormal) < 0.0f;
    if (!outside)
    {
        faceNormal = -faceNormal;
    }

    // Blend the vertex normals for smooth shading, keep it on the face's side
    normal = glm::normalize((1.0f - hitU - hitV) * tri.n0 + hitU * tri.n1 + hitV * tri.n2);
    if (glm::dot(normal, faceNormal) < 0.0f)
    {
        normal = -normal;
    }

    return tClosest;
}
