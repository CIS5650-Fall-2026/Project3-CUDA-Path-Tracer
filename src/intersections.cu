#include "intersections.h"

#include "features.h"

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

// Tests one triangle and, if the ray hits it closer than the best hit so far,
// makes it the new best hit.
__host__ __device__ inline void testTriangle(
    const Triangle* triangles,
    int index,
    const Ray& q,
    float& t_min,
    int& hit_tri,
    float& hit_u,
    float& hit_v)
{
    const Triangle& tri = triangles[index];

    glm::vec3 uvt;
    if (glm::intersectRayTriangle(q.origin, q.direction, tri.v0, tri.v1, tri.v2, uvt) && uvt.z > 0.0f && uvt.z < t_min) {
        hit_tri = index;
        hit_u = uvt.x;
        hit_v = uvt.y;
        t_min = uvt.z;
    }
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    DeviceBVH bvh,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_min = FLT_MAX;
    int hit_tri = -1;
    float hit_u = 0.0f;
    float hit_v = 0.0f;

#if BVH
    if (mesh.bvhRoot < 0) {
        return -1;
    }

    glm::vec3 invDirection = 1.0f / q.direction;

    // Nodes still to visit, with the distance at which the ray enters each so
    // that one made redundant by a closer hit found since can be dropped
    // without re-reading it. Popping a node pushes at most its two children, so
    // this holds at most one waiting sibling per level plus the two just pushed.
    int stackNode[BVH_MAX_DEPTH + 1];
    float stackEntry[BVH_MAX_DEPTH + 1];
    int sp = 0;

    // The root's box is the mesh's bounding box, so this is the same early-out
    // MESH_BOUNDING_VOLUME_CULLING does.
    const BVHNode& root = bvh.nodes[mesh.bvhRoot];
    float rootEntry = aabbEntryDistance(root.bboxMin, root.bboxMax, q.origin, invDirection, t_min);
    if (rootEntry < FLT_MAX) {
        stackNode[sp] = mesh.bvhRoot;
        stackEntry[sp] = rootEntry;
        ++sp;
    }

    while (sp > 0) {
        --sp;
        if (stackEntry[sp] >= t_min) {
            continue;
        }
        const BVHNode& node = bvh.nodes[stackNode[sp]];

        if (node.triCount > 0) {
            for (int i = node.leftOrFirst; i < node.leftOrFirst + node.triCount; ++i) {
                testTriangle(bvh.triangles, i, q, t_min, hit_tri, hit_u, hit_v);
            }
            continue;
        }

        int nearChild = node.leftOrFirst;
        int farChild = node.leftOrFirst + 1;
        const BVHNode& left = bvh.nodes[nearChild];
        const BVHNode& right = bvh.nodes[farChild];
        float nearEntry = aabbEntryDistance(left.bboxMin, left.bboxMax, q.origin, invDirection, t_min);
        float farEntry = aabbEntryDistance(right.bboxMin, right.bboxMax, q.origin, invDirection, t_min);
        if (farEntry < nearEntry) {
            int tmpChild = nearChild;
            nearChild = farChild;
            farChild = tmpChild;
            float tmpEntry = nearEntry;
            nearEntry = farEntry;
            farEntry = tmpEntry;
        }

        // Far child goes on first so the near one is popped next: finding a
        // close hit early lets the far one be skipped.
        if (farEntry < FLT_MAX) {
            stackNode[sp] = farChild;
            stackEntry[sp] = farEntry;
            ++sp;
        }
        if (nearEntry < FLT_MAX) {
            stackNode[sp] = nearChild;
            stackEntry[sp] = nearEntry;
            ++sp;
        }
    }
#else
#if MESH_BOUNDING_VOLUME_CULLING
    // One slab test instead of mesh.triangleCount triangle tests, for every ray
    // that never comes near the mesh.
    if (!aabbIntersectionTest(mesh.bboxMin, mesh.bboxMax, q)) {
        return -1;
    }
#endif

    for (int i = mesh.triangleStart; i < mesh.triangleStart + mesh.triangleCount; ++i) {
        testTriangle(bvh.triangles, i, q, t_min, hit_tri, hit_u, hit_v);
    }
#endif

    if (hit_tri < 0) {
        return -1;
    }

    const Triangle& tri = bvh.triangles[hit_tri];

    glm::vec3 objspaceNormal = glm::normalize((1.0f - hit_u - hit_v) * tri.n0 + hit_u * tri.n1 + hit_v * tri.n2);

    glm::vec3 faceNormal = glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
    outside = glm::dot(q.direction, faceNormal) < 0.0f;
    if (!outside) {
        objspaceNormal = -objspaceNormal;
    }

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(q, t_min), 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objspaceNormal, 0.0f)));

    return glm::length(r.origin - intersectionPoint);
}
