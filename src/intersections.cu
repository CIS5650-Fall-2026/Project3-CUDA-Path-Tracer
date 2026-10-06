#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // transform the ray into box's local space 
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
            // intersection with slabs
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            // update latest entry
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            // update earliest exit
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    // inside the box only when loop through all axes and tmax still >= tmin
    // also reject geometry behind ray start
    if (tmax >= tmin && tmax > 0) 
    {
        outside = true;
        if (tmin <= 0) // no entrance = started inside geometry 
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
    const Triangle &tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    float &u,
    float &v
) {
    // barycentric coordinates - representing a point as a weighted combination of vertices 
    glm::vec3 v0 = tri.positions[0];
    glm::vec3 v1 = tri.positions[1];
    glm::vec3 v2 = tri.positions[2];

    glm::vec3 e0 = v1 - v0;
    glm::vec3 e1 = v2 - v0;

    glm::vec3 dir = r.direction;

    glm::vec3 pVec = glm::cross(dir, e1);
    float det = glm::dot(pVec, e0);

#if CULLING
    // if determinant is negative, triangle is back-facing
    // if determinant is close to 0, ray misses the triangle (parallel)
    if (det < EPSILON) return -1;
#endif

    if (abs(det) < EPSILON) return false;

    float invDet = 1 / det;

    // do barycentric coordinates actually fall inside the traingle? 
    glm::vec3 tVec = r.origin - v0;
    u = glm::dot(tVec, pVec) * invDet;
    if (u < 0 || u > 1) return -1;

    glm::vec3 qVec = glm::cross(tVec, e0);
    v = glm::dot(dir, qVec) * invDet;
    if (v < 0 || u + v > 1) return -1;

    float t = glm::dot(e1, qVec) * invDet;

    if (t < EPSILON) return -1;

    intersectionPoint = r.origin + t * r.direction;
    normal = tri.normal;
    // started from inside or outside? depends on normal
    outside = glm::dot(normal, r.direction) < 0.0f;

    return t;
}
