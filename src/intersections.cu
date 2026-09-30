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
    Triangle tri,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    glm::vec3 temp;
    bool hit = glm::intersectRayTriangle(r.origin, r.direction, tri.v0, tri.v1, tri.v2, temp);

    if (!hit || temp.z <= 0.0f)
    {
        return -1;
    }

    float t = temp.z;
    float w1 = temp.x;
    float w2 = temp.y;
    float w0 = 1.0f - w1 - w2;
    uv = w0 * tri.uv0 + w1 * tri.uv1 + w2 * tri.uv2;
    intersectionPoint = r.origin + t * r.direction;
    normal = tri.normal;
    outside = glm::dot(r.direction, tri.normal) < 0.0f;

    return t;
}

__host__ __device__ bool meshBoundIntersectionTest(
    Ray r,
    glm::vec3 boundsMin,
    glm::vec3 boundsMax)
{
    float closestEntry = -1e38f;
    float closestExit  =  1e38f;

    for (int axis = 0; axis < 3; axis++)
    {
        float tToMin = (boundsMin[axis] - r.origin[axis]) / r.direction[axis];
        float tToMax = (boundsMax[axis] - r.origin[axis]) / r.direction[axis];
        float entry = glm::min(tToMin, tToMax);
        float exit  = glm::max(tToMin, tToMax);
        closestEntry = glm::max(closestEntry, entry);
        closestExit  = glm::min(closestExit, exit);
        if (closestEntry > closestExit)
        {
            return false;
        }
    }
    return closestExit > 0.0f;
}

__host__ __device__ float betterMeshIntersectionTest(
    node* nodes,
    Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    int& hitMaterialId)
{
    float t_min = FLT_MAX;
    bool hitAnything = false;
    int stack[32];
    int stackPtr = 0;
    stack[stackPtr] = 0;
    stackPtr++;
    // check through binary tree, add affected children to stack
    while (stackPtr > 0)
    {
        stackPtr--;
        int nodeIdx = stack[stackPtr];
        node currentNode = nodes[nodeIdx];
        if (!meshBoundIntersectionTest(r, currentNode.minBound, currentNode.maxBound))
        {
            continue;
        }
        // in bound
        if (currentNode.leftChild == -1) // leaf
        {
            for (int i = 0; i < currentNode.triCount; i++)
            {
                Triangle& tri = triangles[currentNode.triStart + i];
                glm::vec3 tmp_intersect, tmp_normal;
                glm::vec2 tmp_uv;
                bool outside;
                float t = triangleIntersectionTest(tri, r, tmp_intersect, tmp_normal, tmp_uv, outside);
                // update values if it's closer
                if (t > 0.0f && t < t_min)
                {
                    t_min = t;
                    intersectionPoint = tmp_intersect;
                    normal = tmp_normal;
                    uv = tmp_uv;
                    hitMaterialId = tri.materialid;
                    hitAnything = true;
                }
            }
        }
        else // not a leaf, add children to stack
        {
            stack[stackPtr++] = currentNode.leftChild;
            stack[stackPtr++] = currentNode.rightChild;
        }
    }
    if (hitAnything) {
        return t_min;
    }
    return -1.0f;
}