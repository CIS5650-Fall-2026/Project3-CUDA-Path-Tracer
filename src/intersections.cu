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
    const Triangle& triangle,
    const Ray& ray,
    glm::vec2& barycentric)
{
    glm::vec3 res;

    //fills in our result where x and y are the ratio of the area for the subtriangles and z is our t val
    bool hit = glm::intersectRayTriangle(
        ray.origin,
        ray.direction,
        triangle.positions[0],
        triangle.positions[1],
        triangle.positions[2],
        res
    );

    //if we miss from the "front" check the back. Needed if we have like transparent materials like glass
    if (!hit) {
        //we reverse the positions as a way to check back sided intersections
        hit = glm::intersectRayTriangle(
            ray.origin,
            ray.direction,
            triangle.positions[0], 
            triangle.positions[2], 
            triangle.positions[1], 
            res
        );

        //since we swapped positions for the intersection we swap back to our original convention 
        if (hit) {
            float temporary = res.x;
            res.x = res.y;
            res.y = temporary;
        }
    }

    //check if we just don't hit the triangle or it is behind us
    if (!hit || res.z <= 0.0f) {
        return -1.0f;
    }

    barycentric = glm::vec2(res.x, res.y);
    return res.z;
}

__host__ __device__ bool boundingBoxIntersectionTest(
    const glm::vec3& minCorner,
    const glm::vec3& maxCorner,
    const Ray& ray,
    float t)
{
    float tEnter = 0.0f;
    float tExit = t;

    for (int axis = 0; axis < 3; axis++) {
        float o = ray.origin[axis];
        float dir = ray.direction[axis];
        float min = minCorner[axis];
        float max = maxCorner[axis];

        //quick check for if that component is out of bounds
        if (dir == 0.0f) {
            if (o < min || o > max)
                return false;
            continue;
        }

        float t0 = (min - o)/dir;
        float t1 = (max - o)/dir;

        //negative directions reverse entry and exit
        if (t0 > t1) {
            float temporary = t0;
            t0 = t1;
            t1 = temporary;
        }

        tEnter = glm::max(tEnter, t0);
        tExit = glm::min(tExit, t1);

        if (tEnter > tExit) 
            return false;
        
    }

    return true;
}