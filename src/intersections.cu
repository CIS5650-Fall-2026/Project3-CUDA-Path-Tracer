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

__host__ __device__ inline bool aabbHit(const AABB& b, const glm::vec3& ro,
    const glm::vec3& invD, float tMax, float& tNear)
{
    glm::vec3 t0 = (b.minBound - ro) * invD;
    glm::vec3 t1 = (b.maxBound - ro) * invD;
    glm::vec3 lo = glm::min(t0, t1), hi = glm::max(t0, t1);
    tNear = fmaxf(fmaxf(lo.x, lo.y), fmaxf(lo.z, 0.f));
    float tFar = fminf(fminf(hi.x, hi.y), fminf(hi.z, tMax));
    return tNear <= tFar;
}

__host__ __device__ inline void triTest(const Triangle* tris, int i, const glm::vec3& ro,
    const glm::vec3& rd, float& bestT, int& bestTri, glm::vec2& bestUV)
{
    glm::vec3 bary;
    if (glm::intersectRayTriangle(ro, rd, tris[i].p0, tris[i].p1, tris[i].p2, bary)
        && bary.z > 1e-4f && bary.z < bestT) {
        bestT = bary.z; bestTri = i; bestUV = glm::vec2(bary.x, bary.y);
    }
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh, Ray r, const Triangle* tris, const BVHNode* nodes, int accelMode,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.f)));
    glm::vec3 invD = 1.f / rd;

    float bestT = 1e30f; int bestTri = -1; glm::vec2 uv(0.f);
    float tn;

    if (accelMode == 2 && mesh.bvhRoot >= 0) {
        int stack[64]; int sp = 0;
        stack[sp++] = mesh.bvhRoot;
        while (sp > 0) {
            const BVHNode& nd = nodes[stack[--sp]];
            if (!aabbHit(nd.bounds, ro, invD, bestT, tn)) continue;
            if (nd.triCount > 0) {
                for (int i = nd.firstTri; i < nd.firstTri + nd.triCount; ++i)
                    triTest(tris, i, ro, rd, bestT, bestTri, uv);
            }
            else {
                float tl, tr;
                bool hl = aabbHit(nodes[nd.left].bounds, ro, invD, bestT, tl);
                bool hr = aabbHit(nodes[nd.right].bounds, ro, invD, bestT, tr);
                if (hl && hr) { // push far child first so near child is popped first
                    stack[sp++] = (tl < tr) ? nd.right : nd.left;
                    stack[sp++] = (tl < tr) ? nd.left : nd.right;
                }
                else if (hl) stack[sp++] = nd.left;
                else if (hr)   stack[sp++] = nd.right;
            }
        }
    }
    else {
        if (accelMode == 1 && !aabbHit(mesh.boundingBox, ro, invD, bestT, tn)) return -1;
        for (int i = mesh.triangleStartIdx; i < mesh.triangleStartIdx + mesh.triangleCount; ++i)
            triTest(tris, i, ro, rd, bestT, bestTri, uv);
    }
    if (bestTri < 0) return -1;

    const Triangle& t = tris[bestTri];
    glm::vec3 gn = glm::cross(t.p1 - t.p0, t.p2 - t.p0);
    outside = glm::dot(gn, rd) < 0.f;
    glm::vec3 n = (1.f - uv.x - uv.y) * t.n0 + uv.x * t.n1 + uv.y * t.n2;
    if (!outside) n = -n;

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(ro + bestT * rd, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(n, 0.f)));
    return glm::length(r.origin - intersectionPoint);
}