#include "intersections.h"

#include "octree.h"

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

__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    //moller-Trumbore
    const float EPS = 1e-8f;
    glm::vec3 e1 = tri.v1 - tri.v0;
    glm::vec3 e2 = tri.v2 - tri.v0;
    glm::vec3 p = glm::cross(r.direction, e2);
    float det = glm::dot(e1, p);
    if (fabsf(det) < EPS)
    {
        return -1;
    }
    float invDet = 1.0f / det;

    glm::vec3 s = r.origin - tri.v0;
    float u = glm::dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return -1;
    }

    glm::vec3 q = glm::cross(s, e1);
    float v = glm::dot(r.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1;
    }

    float t = glm::dot(e2, q) * invDet;
    if (t <= 1e-4f)
    {
        return -1;
    }

    intersectionPoint = r.origin + t * r.direction;
    normal = glm::normalize((1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2);
    outside = glm::dot(normal, r.direction) < 0.0f;
    if (!outside)
    {
        normal = -normal;
    }
    return t;
}

__host__ __device__ bool aabbIntersectionTest(
    const glm::vec3& bboxMin,
    const glm::vec3& bboxMax,
    Ray r,
    float tMax)
{
    float tNear = 0.0f;
    float tFar = tMax;
    for (int axis = 0; axis < 3; ++axis)
    {
        float invD = 1.0f / r.direction[axis];
        float t0 = (bboxMin[axis] - r.origin[axis]) * invD;
        float t1 = (bboxMax[axis] - r.origin[axis]) * invD;
        if (invD < 0.0f)
        {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tNear = t0 > tNear ? t0 : tNear;
        tFar = t1 < tFar ? t1 : tFar;
        if (tFar < tNear)
        {
            return false;
        }
    }
    return true;
}

__host__ __device__ float octreeIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    const OctNode* octreeNodes,
    const int* octreeTriIndices,
    Ray r,
    float tMax,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // Children are pushed so the octant the ray enters first is popped first;
    // once a hit is found, closest shrinks and farther boxes fail the AABB test.
    int nearMask = (r.direction.x < 0.0f ? 1 : 0)
        | (r.direction.y < 0.0f ? 2 : 0)
        | (r.direction.z < 0.0f ? 4 : 0);

    int stack[OCTREE_STACK_SIZE];
    int stackSize = 0;
    stack[stackSize++] = mesh.octreeRoot;

    float tMin = -1;
    float closest = tMax;
    glm::vec3 tmpPoint;
    glm::vec3 tmpNormal;
    bool tmpOutside;

    while (stackSize > 0)
    {
        const OctNode& node = octreeNodes[stack[--stackSize]];
        if (!aabbIntersectionTest(node.bboxMin, node.bboxMax, r, closest))
        {
            continue;
        }

        if (node.firstChild < 0)
        {
            for (int i = node.triStart; i < node.triStart + node.triCount; ++i)
            {
                float t = triangleIntersectionTest(
                    triangles[octreeTriIndices[i]], r, tmpPoint, tmpNormal, tmpOutside);
                if (t > 0.0f && t < closest)
                {
                    closest = t;
                    tMin = t;
                    intersectionPoint = tmpPoint;
                    normal = tmpNormal;
                    outside = tmpOutside;
                }
            }
            continue;
        }

        for (int c = 7; c >= 0; --c)
        {
            int child = node.firstChild + (c ^ nearMask);
            const OctNode& childNode = octreeNodes[child];
            if (childNode.firstChild < 0 && childNode.triCount == 0)
            {
                continue;
            }
            if (stackSize < OCTREE_STACK_SIZE)
            {
                stack[stackSize++] = child;
            }
        }
    }
    return tMin;
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    const OctNode* octreeNodes,
    const int* octreeTriIndices,
    Ray r,
    float tMax,
    int accelMode,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    if (accelMode == MESH_ACCEL_OCTREE && mesh.octreeRoot >= 0)
    {
        return octreeIntersectionTest(mesh, triangles, octreeNodes, octreeTriIndices,
            r, tMax, intersectionPoint, normal, outside);
    }

    if (accelMode != MESH_ACCEL_NONE && !aabbIntersectionTest(mesh.bboxMin, mesh.bboxMax, r, tMax))
    {
        return -1;
    }

    float tMin = -1;
    glm::vec3 tmpPoint;
    glm::vec3 tmpNormal;
    bool tmpOutside;
    for (int i = 0; i < mesh.triangleCount; ++i)
    {
        float t = triangleIntersectionTest(
            triangles[mesh.triangleStart + i], r, tmpPoint, tmpNormal, tmpOutside);
        if (t > 0.0f && (tMin < 0.0f || t < tMin))
        {
            tMin = t;
            intersectionPoint = tmpPoint;
            normal = tmpNormal;
            outside = tmpOutside;
        }
    }
    return tMin;
}
