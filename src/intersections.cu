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
    glm::vec3 tmin_n(0.0f);
    glm::vec3 tmax_n(0.0f);
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            if (glm::abs(qdxyz) < 0.00001f)
            {
                if (q.origin[xyz] < -0.5f || q.origin[xyz] > 0.5f)
                {
                    return -1.0f;
                }
                continue;
            }
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n(0.0f);
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = -n;
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
        intersectionPoint = multiplyMV(box.transform, glm::vec4(q.origin + tmin * q.direction, 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        if (!outside)
        {
            normal = -normal;
        }
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

    glm::vec3 objspaceIntersection = rt.origin + t * rt.direction;

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ static float proceduralDistance(GeomType type, glm::vec3 p)
{
    if (type == TORUS)
    {
        glm::vec2 ring(glm::length(glm::vec2(p.x, p.z)) - 0.32f, p.y);
        return glm::length(ring) - 0.11f;
    }
    if (type == WOVEN_RING)
    {
        float angle = atan2f(p.z, p.x);
        float radialCenter = 0.31f + 0.075f * cosf(3.0f * angle);
        float verticalCenter = 0.12f * sinf(3.0f * angle);
        glm::vec2 tube(glm::length(glm::vec2(p.x, p.z)) - radialCenter,
            p.y - verticalCenter);
        return glm::length(tube) - 0.07f;
    }

    float gyroid = sinf(13.0f * p.x) * cosf(13.0f * p.y)
        + sinf(13.0f * p.y) * cosf(13.0f * p.z)
        + sinf(13.0f * p.z) * cosf(13.0f * p.x);
    float shell = fabsf(gyroid) / 23.0f - 0.045f;
    return glm::max(shell, glm::length(p) - 0.48f);
}

__host__ __device__ float proceduralIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 direction = glm::normalize(
        multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float b = glm::dot(origin, direction);
    float c = glm::dot(origin, origin) - 0.55f * 0.55f;
    float discriminant = b * b - c;
    if (discriminant < 0.0f) return -1.0f;

    float root = sqrtf(discriminant);
    float t = glm::max(0.0f, -b - root);
    float end = -b + root;
    for (int step = 0; step < 160 && t <= end; ++step)
    {
        glm::vec3 p = origin + t * direction;
        float distance = proceduralDistance(geom.type, p);
        if (fabsf(distance) < 0.0008f)
        {
            const float delta = 0.001f;
            glm::vec3 gradient(
                proceduralDistance(geom.type, p + glm::vec3(delta, 0, 0)) - proceduralDistance(geom.type, p - glm::vec3(delta, 0, 0)),
                proceduralDistance(geom.type, p + glm::vec3(0, delta, 0)) - proceduralDistance(geom.type, p - glm::vec3(0, delta, 0)),
                proceduralDistance(geom.type, p + glm::vec3(0, 0, delta)) - proceduralDistance(geom.type, p - glm::vec3(0, 0, delta)));
            if (glm::dot(gradient, gradient) < 1e-12f) return -1.0f;
            intersectionPoint = multiplyMV(geom.transform, glm::vec4(p, 1.0f));
            normal = glm::normalize(multiplyMV(geom.invTranspose,
                glm::vec4(glm::normalize(gradient), 0.0f)));
            outside = glm::dot(r.direction, normal) < 0.0f;
            if (!outside) normal = -normal;
            float worldDistance = glm::length(r.origin - intersectionPoint);
            return worldDistance > 0.0001f ? worldDistance : -1.0f;
        }
        t += glm::max(fabsf(distance) * 0.6f, 0.0008f);
    }
    return -1.0f;
}
