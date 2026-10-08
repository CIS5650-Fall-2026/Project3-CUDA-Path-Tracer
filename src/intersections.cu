#include "intersections.h"
#include "Volume/volume.h"
#include "Volume/vdb_usage.h"
#include "sceneStructs.h"

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

__host__ __device__ glm::vec2 alignedBbxIntersectionTest(glm::vec3 ro, glm::vec3 rd, glm::vec3 min, glm::vec3 max) {
    glm::vec3 ts1 = (min - ro) / rd;
    glm::vec3 ts2 = (max - ro) / rd;
    
    float tEnter = glm::max(glm::max(glm::min(ts1.x, ts2.x), glm::min(ts1.y, ts2.y)), glm::min(ts1.z, ts2.z));
    float tExit = glm::min(glm::min(glm::max(ts1.x, ts2.x), glm::max(ts1.y, ts2.y)), glm::max(ts1.z, ts2.z));

    if (tExit < tEnter || tExit < 0.0f)
        return glm::vec2(-1.0f, -1.0f);

    return glm::vec2(tEnter, tExit);
}

__host__ __device__ glm::vec2 bbxIntersectionTest(Ray r, glm::mat4 inverseTransform) {
    Ray q;
    q.origin    = multiplyMV(inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = multiplyMV(inverseTransform, glm::vec4(r.direction, 0.0f));

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
        return glm::vec2(tmin, tmax);
    }

    return glm::vec2(-1.0f);
}

__host__ __device__ float planeIntersectionTest(Geom plane, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal) {
    Ray q;
    q.direction = multiplyMV(plane.inverseTransform, glm::vec4(r.direction, 0.0f));
    q.origin = multiplyMV(plane.inverseTransform, glm::vec4(r.origin, 1.0f));

    float t = -q.origin.z / q.direction.z;

    glm::vec3 lHit = q.origin + q.direction * t;
    glm::vec3 lNormal = glm::vec3(0.0f,0.0f,1.0f);

    if(max(abs(lHit.x), abs(lHit.y)) > 0.5) // TODO: max(lHit.x*lHit.x, lHit.y*lHit.y) > 0.25
        return -1;

    intersectionPoint = multiplyMV(plane.transform, glm::vec4(lHit, 1.0f));
    normal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(lNormal, 0.0f)));

    return t;
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

__host__ __device__ void sceneIntersectionTest(Geom* geoms, VolumeData& vd, Ray r, int geoms_size, thrust::default_random_engine& rng, int& hit_geom_index, float& t_min, glm::vec3& intersect_point, glm::vec3& normal, bool& hitVolume) {
    float t;
    t_min = FLT_MAX;
    bool outside = true;
    hitVolume = false;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
        } else if(geom.type == PLANE) 
        {
            t = planeIntersectionTest(geom, r, tmp_intersect, tmp_normal);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }

    for(int i=0; i<vd.num_volumes; ++i) {
        const Volume& v = vd.volumes[i];

#if 1
        float t = vdbIntersectionTest(vd, v, r, rng, t_min);//volumeIntersectionTest(r, v, rng, tStart, tEnd);
        if (t > 0.0) {
            // Hit the volume
            hitVolume = true;
            t_min = t;
            hit_geom_index = i;
        }
#else
        glm::vec2 bbxTs = bbxIntersectionTest(r, v.invTransform * v.userInvTransform);
        if(bbxTs.x < bbxTs.y) {
            float tStart = max(0.0f, bbxTs.x);
            float tEnd = min(t_min, bbxTs.y);

            float t = vdbIntersectionTest(density, v, r, rng, tEnd);//volumeIntersectionTest(r, v, rng, tStart, tEnd);
            if(t > 0.0) {
                // Hit the volume
                hitVolume = true;
                t_min = t;
                hit_geom_index = i;
            }
        }

#endif
    }

    
}