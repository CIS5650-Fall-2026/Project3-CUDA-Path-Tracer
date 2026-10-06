#include "intersections.h"
#include <cfloat>

__host__ __device__ float boxIntersectionTest(
    const Geom& box,
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
    const Geom& sphere,
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



__host__ __device__ bool aabbIntersectionTest(glm::vec3 bmin, glm::vec3 bmax, const Ray& r)
{
    glm::vec3 invD = 1.0f / r.direction;
    glm::vec3 t0 = (bmin - r.origin) * invD;
    glm::vec3 t1 = (bmax - r.origin) * invD;
    glm::vec3 tNearV = glm::min(t0, t1);
    glm::vec3 tFarV = glm::max(t0, t1);
    float tNear = fmaxf(fmaxf(tNearV.x, tNearV.y), tNearV.z);
    float tFar = fminf(fminf(tFarV.x, tFarV.y), tFarV.z);
    return tFar >= fmaxf(tNear, 0.0f);
}

__host__ __device__ float triangleIntersectionTest(const Triangle& tri, const Ray& r, float& u, float& v)
{
    glm::vec3 e1 = tri.v1 - tri.v0;
    glm::vec3 e2 = tri.v2 - tri.v0;
    glm::vec3 p = glm::cross(r.direction, e2);
    float det = glm::dot(e1, p);
    if (fabsf(det) < 1e-12f) return -1.0f;   // 光线和三角形平行
    float invDet = 1.0f / det;

    glm::vec3 s = r.origin - tri.v0;
    u = glm::dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f) return -1.0f;

    glm::vec3 q = glm::cross(s, e1);
    v = glm::dot(r.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float t = glm::dot(e2, q) * invDet;
    return t > 0.0f ? t : -1.0f;
}

// Distance along the ray to the box, or FLT_MAX if the ray misses it
// or the box starts beyond tMax (the closest hit found so far)
__host__ __device__ float aabbHitDistance(glm::vec3 bmin, glm::vec3 bmax, glm::vec3 origin, glm::vec3 invD, float tMax)
{
    glm::vec3 t0 = (bmin - origin) * invD;
    glm::vec3 t1 = (bmax - origin) * invD;
    glm::vec3 tNearV = glm::min(t0, t1);
    glm::vec3 tFarV = glm::max(t0, t1);
    float tNear = fmaxf(fmaxf(tNearV.x, tNearV.y), tNearV.z);
    float tFar = fminf(fminf(tFarV.x, tFarV.y), tFarV.z);
    if (tFar < fmaxf(tNear, 0.0f) || tNear > tMax) return FLT_MAX;
    return fmaxf(tNear, 0.0f);
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    const BVHNode* nodes,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    glm::vec3& tangent,
    glm::vec3& bitangent)
{
    float tMin = FLT_MAX;
    int hit = -1;   // absolute triangle index
    float hitU = 0.0f, hitV = 0.0f;

#if USE_BVH
    glm::vec3 invD = 1.0f / r.direction;

    // Iterative traversal with a per-thread stack (no recursion on the GPU)
    int stackNode[64];
    float stackDist[64];
    int sp = 0;

    float dRoot = aabbHitDistance(nodes[mesh.bvhRoot].bboxMin, nodes[mesh.bvhRoot].bboxMax, r.origin, invD, tMin);
    if (dRoot == FLT_MAX) return -1.0f;
    stackNode[sp] = mesh.bvhRoot;
    stackDist[sp] = dRoot;
    sp++;

    while (sp > 0)
    {
        sp--;
        if (stackDist[sp] > tMin) continue;   // a closer hit was found after this node was pushed
        const BVHNode& node = nodes[stackNode[sp]];

        if (node.triCount > 0)   // leaf
        {
            for (int i = node.leftOrFirst; i < node.leftOrFirst + node.triCount; i++)
            {
                float u, v;
                float t = triangleIntersectionTest(triangles[i], r, u, v);
                if (t > 0.0f && t < tMin)
                {
                    tMin = t;
                    hit = i;
                    hitU = u;
                    hitV = v;
                }
            }
        }
        else   // interior: push the farther child first so the nearer one is visited first
        {
            int a = node.leftOrFirst;
            int b = a + 1;
            float dA = aabbHitDistance(nodes[a].bboxMin, nodes[a].bboxMax, r.origin, invD, tMin);
            float dB = aabbHitDistance(nodes[b].bboxMin, nodes[b].bboxMax, r.origin, invD, tMin);
            if (dA > dB)
            {
                int ti = a; a = b; b = ti;
                float tf = dA; dA = dB; dB = tf;
            }
            if (dB != FLT_MAX) { stackNode[sp] = b; stackDist[sp] = dB; sp++; }
            if (dA != FLT_MAX) { stackNode[sp] = a; stackDist[sp] = dA; sp++; }
        }
    }
#else
#if MESH_BBOX_CULLING
    if (!aabbIntersectionTest(mesh.bboxMin, mesh.bboxMax, r)) return -1.0f;
#endif
    for (int i = mesh.triStart; i < mesh.triStart + mesh.triCount; i++)
    {
        float u, v;
        float t = triangleIntersectionTest(triangles[i], r, u, v);
        if (t > 0.0f && t < tMin)
        {
            tMin = t;
            hit = i;
            hitU = u;
            hitV = v;
        }
    }
#endif

    if (hit < 0) return -1.0f;

    const Triangle& tri = triangles[hit];
    intersectionPoint = r.origin + tMin * r.direction;

    // Smooth shading normal from the vertex normals
    glm::vec3 n = glm::normalize((1.0f - hitU - hitV) * tri.n0 + hitU * tri.n1 + hitV * tri.n2);
    uv = (1.0f - hitU - hitV) * tri.t0 + hitU * tri.t1 + hitV * tri.t2;
    tangent = tri.tangent;       // already world space
    bitangent = tri.bitangent;

    // Geometric normal (winding order) decides inside vs outside
    glm::vec3 geoN = glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
    outside = glm::dot(r.direction, geoN) < 0.0f;
    normal = outside ? n : -n;
    return tMin;
}