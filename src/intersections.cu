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

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool boundingBoxTest(
    glm::vec3 rayOrigin,
    glm::vec3 rayDirection,
    glm::vec3 bboxMin,
    glm::vec3 bboxMax,
    float maxT = FLT_MAX) {  // ignore parts of the box farther than maxT (e.g. behind a closer hit)
    float tEnter = 0.0f;
    float tExit = maxT;
    for (int axis = 0; axis < 3; ++axis) {
        float invDir = 1.0f / rayDirection[axis];
        float t1 = (bboxMin[axis] - rayOrigin[axis]) * invDir;
        float t2 = (bboxMax[axis] - rayOrigin[axis]) * invDir;
        tEnter = glm::max(tEnter, glm::min(t1, t2));
        tExit = glm::min(tExit, glm::max(t1, t2));
    }
    return tEnter <= tExit;
}

__host__ __device__ static bool triangleTest(
    glm::vec3 origin, glm::vec3 direction, const Triangle& tri,
    float& t, float& u, float& v) {
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    glm::vec3 p = glm::cross(direction, edge2);
    float det = glm::dot(edge1, p);
    if (glm::abs(det) < 1e-8f) {
        return false;
    }
    float invDet = 1.0f / det;
    glm::vec3 toOrigin = origin - tri.v0;
    u = glm::dot(toOrigin, p) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    glm::vec3 q = glm::cross(toOrigin, edge1);
    v = glm::dot(direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = glm::dot(edge2, q) * invDet;
    return t > 0.0f;
}

struct MeshHit {
    float t;
    int tri;
    float u, v;
};

__host__ __device__ static void testTriangleRange(
    glm::vec3 origin, glm::vec3 direction, const Triangle* triangles,
    int start, int count, MeshHit& hit) {
    for (int i = start; i < start + count; i++) {
        float t, u, v;
        if (triangleTest(origin, direction, triangles[i], t, u, v) && t < hit.t) {
            hit.t = t;
            hit.tri = i;
            hit.u = u;
            hit.v = v;
        }
    }
}

__host__ __device__ static void traverseBoundingVolHier(
    glm::vec3 origin, glm::vec3 direction, const Triangle* triangles,
    const BoundingVolumeHierarchyNode* bvhNodes, int root, MeshHit& hit) {
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = root;
    while (stackSize > 0) {
        const BoundingVolumeHierarchyNode& node = bvhNodes[stack[--stackSize]];
        // skip if the ray misses
        if (!boundingBoxTest(origin, direction, node.bboxMin, node.bboxMax, hit.t)) {
            continue;
        }
        if (node.left == -1) {
            testTriangleRange(origin, direction, triangles, node.start, node.triCount, hit);
        }
        else {
            stack[stackSize++] = node.left;
            stack[stackSize++] = node.right;
        }
    }
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    const Triangle* triangles,
    const BoundingVolumeHierarchyNode* bvhNodes,
    Ray r,
    bool useBoundsCulling,
    bool useBVH,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    MeshHit hit;
    hit.t = FLT_MAX;
    hit.tri = -1;
    hit.u = 0.0f;
    hit.v = 0.0f;
    if (useBVH) {
        traverseBoundingVolHier(origin, direction, triangles, bvhNodes, mesh.bouindingVolumeHierarchyRoot, hit);
    }
    else if (useBoundsCulling && !boundingBoxTest(origin, direction, mesh.bboxMin, mesh.bboxMax)) {
        return -1;
    }
    else {
        testTriangleRange(origin, direction, triangles, mesh.start, mesh.triCount, hit);
    }
    if (hit.tri == -1) {
        return -1;
    }
    int closestTri = hit.tri;
    float closestT = hit.t;
    float closestU = hit.u, closestV = hit.v;
    const Triangle& tri = triangles[closestTri];
    glm::vec3 objectNormal = glm::normalize(
        (1.0f - closestU - closestV) * tri.n0 + closestU * tri.n1 + closestV * tri.n2);
    outside = glm::dot(objectNormal, direction) < 0.0f;
    glm::vec3 objectPoint = origin + closestT * direction;
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objectPoint, 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objectNormal, 0.0f)));
    return glm::length(r.origin - intersectionPoint);
}
