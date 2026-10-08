#include "intersections.h"
#include "sdf.h"

#include <cfloat>

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
                tmax_n = - n;
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

__host__ __device__ float rectIntersectionTest(
    Geom rect,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 ro = multiplyMV(rect.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(rect.inverseTransform, glm::vec4(r.direction, 0.0f));

    if (glm::abs(rd.y) < 1e-8f) return -1;
    float t = (-0.5f - ro.y) / rd.y;
    if (t <= 0.0f) return -1;

    glm::vec3 p = ro + t * rd;
    if (glm::abs(p.x) > 0.5f || glm::abs(p.z) > 0.5f) return -1;

    intersectionPoint = multiplyMV(rect.transform, glm::vec4(p, 1.0f));
    normal = glm::normalize(multiplyMV(rect.invTranspose, glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));
    outside = glm::dot(r.direction, normal) < 0.0f;
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ void findClosestIntersection(
    Ray r,
    Geom* geoms,
    int geoms_size,
    float& t_min,
    glm::vec3& intersect_point,
    glm::vec3& normal,
    int& hit_geom_index,
    Geom* emissiveGeoms,
    int emissive_size)
{
    float t;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    bool outside = true;

    t_min = FLT_MAX;
    hit_geom_index = -1;

    for (int i = 0; i < geoms_size + emissive_size; i++)
    {
        Geom& geom = (i < geoms_size) ? geoms[i] : emissiveGeoms[i - geoms_size];
        t = -1.0f;

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SDF)         {
            t = sdfIntersectionTest(r, tmp_intersect, tmp_normal);
		}
        else if (geom.type == RECT2D)
        {
            t = rectIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }
}

__host__ __device__ float rectArea(const Geom& rect) {
    glm::vec3 e1 = glm::vec3(rect.transform * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    glm::vec3 e2 = glm::vec3(rect.transform * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    return glm::length(glm::cross(e1, e2));
}

__host__ __device__ void sampleRect(const Geom& rect, float u, float v, glm::vec3& pos, glm::vec3& nor) {
	glm::vec4 localPos = glm::vec4(u - 0.5f, -0.5f, v - 0.5f, 1.0f); // -0.5, 0.5
    pos = glm::vec3(rect.transform * localPos);
    nor = glm::normalize(glm::vec3(rect.invTranspose * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));
}
