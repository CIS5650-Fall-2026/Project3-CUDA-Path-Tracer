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
    /*if (!outside)
    {
        normal = -normal;
    }*/

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleArea(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) {
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    return 0.5f * glm::length(glm::cross(e1, e2));
}

__host__ __device__ glm::vec3 barycentricInterpolateNormal(const Geom& triangle, glm::vec3 intersectionPoint) {
    float totalArea = triangleArea(triangle.v0, triangle.v1, triangle.v2);

    float t0Area = triangleArea(intersectionPoint, triangle.v1, triangle.v2);
    float t1Area = triangleArea(triangle.v0, intersectionPoint, triangle.v2);
    float t2Area = triangleArea(triangle.v0, triangle.v1, intersectionPoint);

    return triangle.norm0 * (t0Area / totalArea) + triangle.norm1 * (t1Area / totalArea) + triangle.norm2 * (t2Area / totalArea);
}

// This function uses glm::intersectRayTriangle code converted to CUDA with code from other intersect functions to give standardized intersection data
__host__ __device__ float triangleIntersectionTest(
    Geom triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{   

    glm::vec3 baryPosition;

    glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    glm::vec3 e1 = triangle.v1 - triangle.v0;
	glm::vec3 e2 = triangle.v2 - triangle.v0;

    glm::vec3 p = glm::cross(rt.direction, e2);

    float a = glm::dot(e1, p);
    if(a < FLT_EPSILON) return -1;

    float f = 1.0f / a;

    glm::vec3 s = rt.origin - triangle.v0;
    baryPosition.x = f * glm::dot(s, p);
    if(baryPosition.x < 0.0f) return -1;
	if(baryPosition.x > 1.0f) return -1;

    glm::vec3 q = glm::cross(s, e1);
    baryPosition.y = f * glm::dot(rt.direction, q);
	if(baryPosition.y < 0.0f) return -1;
	if(baryPosition.y + baryPosition.x > 1.0f) return -1;

    baryPosition.z = f * glm::dot(e2, q);
    if (baryPosition.z < 0.0f) return -1;

    intersectionPoint = r.origin + baryPosition.z * r.direction;
    glm::vec3 objspaceNormal;
    if (triangle.norm0 == glm::vec3(0.0f) && triangle.norm1 == glm::vec3(0.0f) && triangle.norm2 == glm::vec3(0.0f)) {
        // normal from cross product
        objspaceNormal = glm::normalize(glm::cross(e1, e2));
    }
    else {
        // normal from interpolating vertex normals from gltf
        objspaceNormal = barycentricInterpolateNormal(triangle, intersectionPoint);
    }
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(objspaceNormal, 0.f)));
    if (glm::dot(normal, r.direction) < FLT_EPSILON) {
        outside = true;
    }
    else {
        outside = false;
    }

    return baryPosition.z;
}

__host__ __device__ bool hitsBoundingBox(
    Geom box,
    Ray r)
{
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    float t = boxIntersectionTest(box, r, tmp_intersect, tmp_normal, outside);
    return t > 0.0f;
}
