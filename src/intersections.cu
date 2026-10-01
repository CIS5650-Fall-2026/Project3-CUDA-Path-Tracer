#include "intersections.h"

__host__ __device__ float torusSDF(glm::vec3 p)
{
    const float majorRadius = 0.35f;
    const float minorRadius = 0.15f;
    glm::vec2 q(glm::length(glm::vec2(p.x, p.z)) - majorRadius, p.y);
    return glm::length(q) - minorRadius;
}

__host__ __device__ glm::vec3 torusNormal(glm::vec3 p)
{
    const float e = 0.001f;

    float dx = torusSDF(p + glm::vec3(e, 0.0f, 0.0f)) - torusSDF(p - glm::vec3(e, 0.0f, 0.0f));
    float dy = torusSDF(p + glm::vec3(0.0f, e, 0.0f)) - torusSDF(p - glm::vec3(0.0f, e, 0.0f));
    float dz = torusSDF(p + glm::vec3(0.0f, 0.0f, e)) - torusSDF(p - glm::vec3(0.0f, 0.0f, e));

    return glm::normalize(glm::vec3(dx, dy, dz));
}

__host__ __device__ float boxSDF(glm::vec3 p, glm::vec3 halfSize)
{
    glm::vec3 q = glm::abs(p) - halfSize;
    glm::vec3 outside(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f));

    float outsideDistance = glm::length(outside);
    float insideDistance = fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);

    return outsideDistance + insideDistance;
}

__host__ __device__ float mengerSDF(glm::vec3 p)
{
    const float holeRadius = 1.0f / 6.0f;

    float outerBox = boxSDF(p, glm::vec3(0.5f));

    float holeX = boxSDF(p, glm::vec3(0.6f, holeRadius, holeRadius));
    float holeY = boxSDF(p, glm::vec3(holeRadius, 0.6f, holeRadius));
    float holeZ = boxSDF(p, glm::vec3(holeRadius, holeRadius, 0.6f));

    float holes = fminf(holeX, fminf(holeY, holeZ));

    return fmaxf(outerBox, -holes);
}

__host__ __device__ glm::vec3 mengerNormal(glm::vec3 p)
{
    const float e = 0.001f;

    float dx = mengerSDF(p + glm::vec3(e, 0.0f, 0.0f)) - mengerSDF(p - glm::vec3(e, 0.0f, 0.0f));
    float dy = mengerSDF(p + glm::vec3(0.0f, e, 0.0f)) - mengerSDF(p - glm::vec3(0.0f, e, 0.0f));
    float dz = mengerSDF(p + glm::vec3(0.0f, 0.0f, e)) - mengerSDF(p - glm::vec3(0.0f, 0.0f, e));

    return glm::normalize(glm::vec3(dx, dy, dz));
}

__host__ __device__ float mengerIntersectionTest(
    Geom menger,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(menger.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(menger.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = 0.001f;

    for (int i = 0; i < 128; i++)
    {
        glm::vec3 p = q.origin + t * q.direction;
        float distance = mengerSDF(p);

        if (fabsf(distance) < 0.001f)
        {
            intersectionPoint = multiplyMV(menger.transform, glm::vec4(p, 1.0f));

            glm::vec3 objectNormal = mengerNormal(p);
            normal = glm::normalize(multiplyMV(menger.invTranspose, glm::vec4(objectNormal, 0.0f)));

            outside = mengerSDF(q.origin) > 0.0f;

            if (!outside)
            {
                normal = -normal;
            }

            return glm::length(r.origin - intersectionPoint);
        }

        t += fabsf(distance);

        if (t > 10.0f)
        {
            break;
        }
    }

    return -1.0f;
}

__host__ __device__ float torusIntersectionTest(
    Geom torus,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(torus.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(torus.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = 0.001f;

    for (int i = 0; i < 128; i++)
    {
        glm::vec3 p = q.origin + t * q.direction;
        float distance = torusSDF(p);

        if (fabsf(distance) < 0.001f)
        {
            intersectionPoint = multiplyMV(torus.transform, glm::vec4(p, 1.0f));

            glm::vec3 objectNormal = torusNormal(p);
            normal = glm::normalize(multiplyMV(torus.invTranspose, glm::vec4(objectNormal, 0.0f)));

            outside = torusSDF(q.origin) > 0.0f;

            if (!outside)
            {
                normal = -normal;
            }

            return glm::length(r.origin - intersectionPoint);
        }

        t += fabsf(distance);

        if (t > 10.0f)
        {
            break;
        }
    }

    return -1.0f;
}

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

__device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& ray,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
    const float EPSILON = 0.000001f;

    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;

    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);

    if (fabsf(a) < EPSILON)
    {
        return -1.0f;
    }

    float f = 1.0f / a;

    glm::vec3 s = ray.origin - triangle.v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
    {
        return -1.0f;
    }

    float t = f * glm::dot(edge2, q);

    if (t <= EPSILON)
    {
        return -1.0f;
    }

    intersectionPoint = ray.origin + t * ray.direction;

    normal = triangle.normal;

    if (glm::dot(normal, ray.direction) > 0.0f)
    {
        normal = -normal;
    }

    return t;
}