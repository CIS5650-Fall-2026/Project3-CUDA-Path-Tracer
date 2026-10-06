#include "intersections.h"
#define MESH_BBOX_CULLING 1
#define MESH_BVH 1

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

    /* handle being inside with the inward normal, flip in refractive case for scatterray */
    //if (!outside)
    //{
    //    normal = -normal;
    //}

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersect(const Ray& rt, const Triangle& tr, float& u, float& v) {
    // find distance to triangle
    auto n = glm::cross(tr.v1 - tr.v0, tr.v2 - tr.v0);
    auto d = glm::dot(tr.v0 - rt.origin, n);
    auto rate = glm::dot(rt.direction, n); // rate approaching normal
    if (fabsf(rate) < 1e-8f) return -1.f; // parallel, will never reach
    auto t = d / rate; // time = distance / speed
    if (t < 1e-4f) return -1.f; // plane behind ray never will reach, with some allowance for being close
    auto p = rt.origin + t * rt.direction;

    // convert to barycentric
    auto e1 = tr.v1 - tr.v0;
    auto e2 = tr.v2 - tr.v0;
    auto vp = p - tr.v0;
    float nn = glm::dot(n, n);

    u = glm::dot(n, glm::cross(vp, e2)) / nn; // weight of v1
    v = glm::dot(n, glm::cross(e1, vp)) / nn; // weight of v2

    if (u < 0.f || v < 0.f || u + v > 1.f) return -1.f;
    return t;
}

/* code reorg putting the test in here */
__host__ __device__ float enterBox(const Ray& r, glm::vec3 invDir, glm::vec3 bmin, glm::vec3 bmax) {
    glm::vec3 t0 = (bmin - r.origin) * invDir;
    glm::vec3 t1 = (bmax - r.origin) * invDir;
    glm::vec3 tEnter = glm::min(t0, t1);
    glm::vec3 tExit = glm::max(t0, t1);
    float tNear = fmaxf(fmaxf(tEnter.x, tEnter.y), tEnter.z); // when have u entered all 3 dims
    float tFar = fminf(fminf(tExit.x, tExit.y), tExit.z); // when have u exited all 3 dims
    if (tFar < fmaxf(tNear, 0.f)) return -1.f; // missed box
    return fmaxf(tNear, 0.f);
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    const Triangle* triangles,   // the global triangle array (dev_triangles)
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    const BVHNode* nodes // the bvhnode array
)
{
    // mesh has object space
    Ray rt;
    rt.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    rt.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

#if MESH_BBOX_CULLING
    {
        if (enterBox(rt, 1.f / rt.direction, mesh.bboxMin, mesh.bboxMax) < 0.f) return -1.f;
    }
#endif
    // defaults if we dont use the BVH
    float tMin = FLT_MAX;
    int hitIndex = -1;
    float hitU = 0.f, hitV = 0.f;
#if MESH_BVH
    // traverse bvh
    auto invDir = 1.f / rt.direction;
    int stack[64];
    int sp = 0;
    stack[sp++] = mesh.root; // push the root


    while (sp > 0) {
        const BVHNode& node = nodes[stack[--sp]]; // pop
        float tBox = enterBox(rt, invDir, node.bottom_corner, node.top_corner);
        if (tBox < 0.f || tBox > tMin) continue; // missed this node no need to explore
        if (node.left_child == -1) {
            // leaf test true
            for (int i = node.t_start_idx; i < node.t_start_idx + node.t_count; ++i) {
                float u, v;
                float t = triangleIntersect(rt, triangles[i], u, v);
                if (t > 0.f && t < tMin) { tMin = t; hitIndex = i; hitU = u; hitV = v; }
            }
        }
        else {
            // push to stack, not leaf has 2 children
            stack[sp++] = node.right_child;
            stack[sp++] = node.left_child;
        }
    }
#else
    // loop over ALL triangles
    for (int i = mesh.triStart; i < mesh.triStart + mesh.triCount; ++i) {
        float u, v;
        float t = triangleIntersect(rt, triangles[i], u, v);
        if (t > 0.f && t < tMin) {
            // closest triangle hit so far
            tMin = t;
            hitIndex = i;
            hitU = u;
            hitV = v;
        }
    }
#endif

    // finished iterating all triangles
    if (hitIndex < 0) return -1.f; // hitindex never updated, all triangles missed
    const auto& tr = triangles[hitIndex];
    auto point = rt.origin + tMin * rt.direction;
    auto objNormal = (1.f - hitU - hitV) * tr.n0 + hitU * tr.n1 + hitV * tr.n2; // blending normals, object space

    // convert to world space
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(point, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objNormal, 0.f)));

    outside = glm::dot(r.direction, normal) < 0.f;

    return glm::length(r.origin - intersectionPoint);
}