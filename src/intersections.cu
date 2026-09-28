#include "intersections.h"

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

__host__ __device__ float triangleIntersectionTest(
    const Triangle &tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // double-sided Moller-Trumbore. glm::intersectRayTriangle culls
    // backfaces, which breaks badly wound meshes and refraction through solids.
    const glm::vec3 edge1 = tri.v1 - tri.v0;
    const glm::vec3 edge2 = tri.v2 - tri.v0;
    const glm::vec3 pvec = glm::cross(r.direction, edge2);
    const float det = glm::dot(edge1, pvec);

    if (fabsf(det) < 1e-9f)
    {
        return -1;
    }

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = r.origin - tri.v0;
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return -1;
    }

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1;
    }

    const float t = glm::dot(edge2, qvec) * invDet;
    if (t <= 0.0f)
    {
        return -1;
    }

    const float w = 1.0f - u - v;

    intersectionPoint = r.origin + t * r.direction;
    glm::vec3 shadingNormal = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);

    // same convention as the analytic prims: normal faces the ray,
    // outside carries the real side
    outside = glm::dot(r.direction, shadingNormal) < 0.0f;
    normal = outside ? shadingNormal : -shadingNormal;

    return t;
}

__host__ __device__ inline bool aabbIntersect(
    const glm::vec3 &bboxMin,
    const glm::vec3 &bboxMax,
    const glm::vec3 &origin,
    const glm::vec3 &invDir,
    float tMax,
    float &tEnter)
{
    glm::vec3 t0 = (bboxMin - origin) * invDir;
    glm::vec3 t1 = (bboxMax - origin) * invDir;
    glm::vec3 tSmall = glm::min(t0, t1);
    glm::vec3 tBig = glm::max(t0, t1);

    float tMin = fmaxf(fmaxf(tSmall.x, tSmall.y), tSmall.z);
    float tFar = fminf(fminf(tBig.x, tBig.y), tBig.z);

    tEnter = fmaxf(tMin, 0.0f);
    return tFar >= tEnter && tEnter < tMax;
}

__host__ __device__ float bvhIntersectionTest(
    const BVHNode *nodes,
    const Triangle *triangles,
    Ray r,
    float tMax,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    int &materialId)
{
    const glm::vec3 dir = glm::normalize(r.direction);
    const glm::vec3 invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    Ray normalizedRay{ r.origin, dir };

    float tBest = tMax;
    bool hit = false;

    // explicit stack, no recursion on the GPU. 64 is plenty for a
    // median-split tree.
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0)
    {
        const BVHNode node = nodes[stack[--stackSize]];

        float tEnter;
        if (!aabbIntersect(node.bboxMin, node.bboxMax, normalizedRay.origin, invDir, tBest, tEnter))
        {
            continue;
        }

        if (node.leftChild < 0)
        {
            for (int i = 0; i < node.triCount; i++)
            {
                const Triangle &tri = triangles[node.firstTri + i];
                glm::vec3 tmpPoint;
                glm::vec3 tmpNormal;
                bool tmpOutside;
                float t = triangleIntersectionTest(tri, normalizedRay, tmpPoint, tmpNormal, tmpOutside);
                if (t > 0.0f && t < tBest)
                {
                    tBest = t;
                    intersectionPoint = tmpPoint;
                    normal = tmpNormal;
                    outside = tmpOutside;
                    materialId = tri.materialId;
                    hit = true;
                }
            }
        }
        else
        {
            stack[stackSize++] = node.leftChild;
            stack[stackSize++] = node.leftChild + 1;
        }
    }

    return hit ? tBest : -1.0f;
}

__host__ __device__ float meshBruteForceIntersectionTest(
    const Triangle *triangles,
    int triangleCount,
    Ray r,
    float tMax,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    int &materialId)
{
    Ray normalizedRay{ r.origin, glm::normalize(r.direction) };

    float tBest = tMax;
    bool hit = false;

    for (int i = 0; i < triangleCount; i++)
    {
        glm::vec3 tmpPoint;
        glm::vec3 tmpNormal;
        bool tmpOutside;
        float t = triangleIntersectionTest(triangles[i], normalizedRay, tmpPoint, tmpNormal, tmpOutside);
        if (t > 0.0f && t < tBest)
        {
            tBest = t;
            intersectionPoint = tmpPoint;
            normal = tmpNormal;
            outside = tmpOutside;
            materialId = triangles[i].materialId;
            hit = true;
        }
    }

    return hit ? tBest : -1.0f;
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
