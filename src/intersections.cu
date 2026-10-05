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
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
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
    const Triangle& triangle,
    Ray r,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 edge1 = triangle.p1 - triangle.p0;
    glm::vec3 edge2 = triangle.p2 - triangle.p0;
    glm::vec3 p = glm::cross(r.direction, edge2);
    float determinant = glm::dot(edge1, p);
    if (abs(determinant) < 0.0000001f) return -1.0f;

    float inverseDeterminant = 1.0f / determinant;
    glm::vec3 t = r.origin - triangle.p0;
    float u = glm::dot(t, p) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f) return -1.0f;

    glm::vec3 q = glm::cross(t, edge1);
    float v = glm::dot(r.direction, q) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float distance = glm::dot(edge2, q) * inverseDeterminant;
    if (distance <= 0.00001f) return -1.0f;

    glm::vec3 geometricNormal = glm::normalize(glm::cross(edge1, edge2));
    normal = glm::normalize((1.0f - u - v) * triangle.n0 + u * triangle.n1 + v * triangle.n2);
    if (glm::dot(normal, geometricNormal) < 0.0f) normal = -normal;
    outside = glm::dot(r.direction, geometricNormal) < 0.0f;
    return distance;
}

__host__ __device__ bool aabbIntersectionTest(
    const Aabb& bounds,
    Ray r,
    float maximumT,
    float& nearT)
{
    float minimumT = 0.0f;
    float farT = maximumT;
    for (int axis = 0; axis < 3; axis++)
    {
        if (abs(r.direction[axis]) < 0.0000001f)
        {
            if (r.origin[axis] < bounds.minimum[axis] || r.origin[axis] > bounds.maximum[axis])
            {
                return false;
            }
            continue;
        }

        float inverseDirection = 1.0f / r.direction[axis];
        float t0 = (bounds.minimum[axis] - r.origin[axis]) * inverseDirection;
        float t1 = (bounds.maximum[axis] - r.origin[axis]) * inverseDirection;
        if (t0 > t1)
        {
            float swap = t0;
            t0 = t1;
            t1 = swap;
        }
        minimumT = max(minimumT, t0);
        farT = min(farT, t1);
        if (farT < minimumT) return false;
    }

    nearT = minimumT;
    return true;
}

__host__ __device__ float primitiveIntersectionTest(
    const Primitive& primitive,
    const Geom* geoms,
    const Triangle* triangles,
    Ray r,
    glm::vec3& normal,
    bool& outside)
{
    if (primitive.type == PRIMITIVE_TRIANGLE)
    {
        return triangleIntersectionTest(triangles[primitive.index], r, normal, outside);
    }

    glm::vec3 intersection(0.0f);
    if (primitive.type == PRIMITIVE_CUBE)
    {
        return boxIntersectionTest(geoms[primitive.index], r, intersection, normal, outside);
    }
    return sphereIntersectionTest(geoms[primitive.index], r, intersection, normal, outside);
}
