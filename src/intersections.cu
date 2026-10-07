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
    Triangle triangle,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    glm::vec2 uv;
    return triangleIntersectionTest(triangle, r, intersectionPoint, normal,
                                    outside, uv);
}

__host__ __device__ float triangleIntersectionTest(
    Triangle triangle,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    glm::vec2 &uv)
{
    glm::vec3 tangent;
    float tangentSign;
    return triangleIntersectionTest(triangle, r, intersectionPoint, normal,
                                    outside, uv, tangent, tangentSign);
}

__host__ __device__ float triangleIntersectionTest(
    Triangle triangle,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    glm::vec2 &uv,
    glm::vec3 &tangent,
    float &tangentSign)
{
    const float epsilon = 0.000001f;
    glm::vec3 direction = glm::normalize(r.direction);
    glm::vec3 baryPosition;
    bool hit = glm::intersectRayTriangle(r.origin, direction, triangle.v0,
                                         triangle.v1, triangle.v2,
                                         baryPosition);
    float u = baryPosition.x;
    float v = baryPosition.y;
    float t = baryPosition.z;
    if (!hit)
    {
        hit = glm::intersectRayTriangle(r.origin, direction, triangle.v0,
                                        triangle.v2, triangle.v1,
                                        baryPosition);
        u = baryPosition.y;
        v = baryPosition.x;
        t = baryPosition.z;
    }

    if (!hit)
    {
        return -1;
    }

    if (t <= epsilon)
    {
        return -1;
    }

    Ray normalizedRay;
    normalizedRay.origin = r.origin;
    normalizedRay.direction = direction;
    intersectionPoint = getPointOnRay(normalizedRay, t);

    float w = 1.0f - u - v;
    uv = w * triangle.uv0 + u * triangle.uv1 + v * triangle.uv2;
    normal = w * triangle.n0 + u * triangle.n1 + v * triangle.n2;
    tangent = w * triangle.t0 + u * triangle.t1 + v * triangle.t2;
    tangentSign = w * triangle.tangentSign0 + u * triangle.tangentSign1 +
                  v * triangle.tangentSign2;
    tangentSign = tangentSign < 0.0f ? -1.0f : 1.0f;
    if (glm::dot(normal, normal) <= epsilon)
    {
        glm::vec3 edge1 = triangle.v1 - triangle.v0;
        glm::vec3 edge2 = triangle.v2 - triangle.v0;
        normal = glm::cross(edge1, edge2);
    }
    normal = glm::normalize(normal);
    if (glm::dot(tangent, tangent) > epsilon)
    {
        tangent = glm::normalize(tangent - normal * glm::dot(normal, tangent));
    }

    outside = glm::dot(normal, direction) < 0.0f;
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}
