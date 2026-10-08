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

__host__ __device__ static float finishMeshHit(
    const Triangle& tri,
    const Ray& r,
    float t,
    const glm::vec2& bary,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside) {
    glm::vec3 geometricNormal = glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
    if (glm::dot(geometricNormal, r.direction) > 0.0f) {
        geometricNormal = -geometricNormal;
    }

    glm::vec3 shadingNormal = (1.0f - bary.x - bary.y) * tri.n0 + bary.x * tri.n1 + bary.y * tri.n2;
    if (!(glm::dot(shadingNormal, shadingNormal) > 0.0f)) {
        shadingNormal = geometricNormal;
    }
    shadingNormal = glm::normalize(shadingNormal);

    outside = glm::dot(shadingNormal, r.direction) < 0.0f;
    if (glm::dot(shadingNormal, geometricNormal) < 0.0f) {
        shadingNormal = -shadingNormal;
    }
    normal = shadingNormal;

    intersectionPoint = getPointOnRay(r, t);
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside) {
    r.direction = glm::normalize(r.direction);

    float tClosest = FLT_MAX;
    int closest = -1;
    glm::vec2 closestBary;
    for (int i = mesh.triangleStart; i < mesh.triangleStart + mesh.triangleCount; i++) {
        glm::vec2 bary;
        const float t = triangleIntersectionTest(triangles[i], r, bary);
        if (t > 0.0f && t < tClosest) {
            tClosest = t;
            closest = i;
            closestBary = bary;
        }
    }

    if (closest < 0) {
        return -1;
    }
    return finishMeshHit(triangles[closest], r, tClosest, closestBary, intersectionPoint, normal, outside);
}

__host__ __device__ float meshIntersectionTestBVH(
    const Geom& mesh,
    const Triangle* triangles,
    const BVHNode* nodes,
    Ray r,
    float tMax,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside) {
    if (mesh.bvhRoot < 0) {
        return -1;
    }

    r.direction = glm::normalize(r.direction);
    const glm::vec3 indvDirection = 1.0f / r.direction;

    const BVHNode& root = nodes[mesh.bvhRoot];
    if (aabbIntersectionTest(root.boundsMin, root.boundsMax, r.origin, indvDirection, tMax) < 0.0f) {
        return -1;
    }

    float tClosest = FLT_MAX;
    int closest = -1;
    glm::vec2 closestBary;

    int pendingNode[BVH_MAX_DEPTH];
    float pendingEntry[BVH_MAX_DEPTH];
    int pending = 0;

    int nodeIndex = mesh.bvhRoot;
    while (true) {
        const BVHNode& node = nodes[nodeIndex];
        if (node.triangleCount > 0 ) {
            for (int i = node.leftOrFirst; i < node.leftOrFirst + node.triangleCount; i++) {
                glm::vec2 bary;
                const float t = triangleIntersectionTest(triangles[i], r, bary);
                if (t > 0.0f && t < tClosest) {
                    tClosest = t;
                    closest = i;
                    closestBary = bary;
                }
            }
        } else {
            const int left = node.leftOrFirst;
            const int right = left + 1;
            const float tLeft = aabbIntersectionTest(nodes[left].boundsMin, nodes[left].boundsMax,
                r.origin, indvDirection, tClosest);
            const float tRight = aabbIntersectionTest(nodes[right].boundsMin, nodes[right].boundsMax,
                r.origin, indvDirection, tClosest);

            if (tLeft >= 0.0f && tRight >= 0.0f) {
                const bool leftFirst = tLeft <= tRight;
                pendingNode[pending] = leftFirst ? right : left;
                pendingEntry[pending] = leftFirst ? tRight : tLeft;
                pending++;
                nodeIndex = leftFirst ? left : right;
                continue;
            }
            if (tLeft >= 0.0f) {
                nodeIndex = left;
                continue;
            }
            if (tRight >= 0.0f) {
                nodeIndex = right;
                continue;
            }
        }

        bool resumed = false;
        while (pending > 0) {
            pending--;
            if (pendingEntry[pending] < tClosest) {
                nodeIndex = pendingNode[pending];
                resumed = true;
                break;
            }
        }
        if (!resumed) {
            break;
        }
    }
    if (closest < 0) {
        return -1;
    }
    return finishMeshHit(triangles[closest], r, tClosest, closestBary, intersectionPoint, normal, outside);
}