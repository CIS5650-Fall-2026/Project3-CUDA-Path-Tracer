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
        if (qdxyz == 0.0f) {
            if (q.origin[xyz] < -0.5f || q.origin[xyz] > 0.5f) {
                return -1.0f;
            }
            continue;
        }
        {
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

__host__ __device__ bool intersectsBounds(Ray ray, glm::vec3 boundsMin,
    glm::vec3 boundsMax, float maxDistance) {
    float nearDistance = 0.0f;
    float farDistance = maxDistance;
    for (int axis = 0; axis < 3; axis++) {
        if (ray.direction[axis] == 0.0f) {
            if (ray.origin[axis] < boundsMin[axis] || ray.origin[axis] > boundsMax[axis]) {
                return false;
            }
            continue;
        }
        float a = (boundsMin[axis] - ray.origin[axis]) / ray.direction[axis];
        float b = (boundsMax[axis] - ray.origin[axis]) / ray.direction[axis];
        nearDistance = glm::max(nearDistance, glm::min(a, b));
        farDistance = glm::min(farDistance, glm::max(a, b));
        if (nearDistance > farDistance) return false;
    }
    return farDistance > 0.0f;
}

__host__ __device__ float triangleIntersectionTest(const Triangle& triangle, Ray ray) {
    glm::vec3 edge1 = triangle.b - triangle.a;
    glm::vec3 edge2 = triangle.c - triangle.a;
    glm::vec3 perpendicular = glm::cross(ray.direction, edge2);
    float determinant = glm::dot(edge1, perpendicular);
    if (fabsf(determinant) < 1e-8f) return -1.0f;
    float inverse = 1.0f / determinant;
    glm::vec3 offset = ray.origin - triangle.a;
    float u = glm::dot(offset, perpendicular) * inverse;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    glm::vec3 crossOffset = glm::cross(offset, edge1);
    float v = glm::dot(ray.direction, crossOffset) * inverse;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    float distance = glm::dot(edge2, crossOffset) * inverse;
    return distance > 0.0f ? distance : -1.0f;
}

__host__ __device__ float meshIntersectionTest(const Geom& mesh, const Triangle* triangles,
    Ray ray, glm::vec3& normal, bool culling, float maxDistance, int* hitTriangle) {
    if (culling && !intersectsBounds(ray, mesh.boundsMin, mesh.boundsMax, maxDistance)) {
        return -1.0f;
    }
    float closest = maxDistance;
    bool hit = false;
    for (int i = 0; i < mesh.triangleCount; i++) {
        const Triangle& triangle = triangles[mesh.triangleStart + i];
        float distance = triangleIntersectionTest(triangle, ray);
        if (distance > 0.0f && distance < closest) {
            closest = distance;
            normal = triangle.normal;
            if (hitTriangle != nullptr) *hitTriangle = mesh.triangleStart + i;
            hit = true;
        }
    }
    return hit ? closest : -1.0f;
}
