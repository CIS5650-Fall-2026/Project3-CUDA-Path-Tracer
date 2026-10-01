#include "intersections.h"

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
            glm::vec3 n(0.0f);
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
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

__host__ __device__ bool rayHitsBounds(
    const Ray& r,
    glm::vec3 boundsMin,
    glm::vec3 boundsMax)
{
    const glm::vec3 invDir = 1.0f / r.direction;
    const glm::vec3 t0 = (boundsMin - r.origin) * invDir;
    const glm::vec3 t1 = (boundsMax - r.origin) * invDir;
    const glm::vec3 tSmall = glm::min(t0, t1);
    const glm::vec3 tBig = glm::max(t0, t1);

    const float tNear = glm::max(glm::max(tSmall.x, tSmall.y), tSmall.z);
    const float tFar = glm::min(glm::min(tBig.x, tBig.y), tBig.z);
    return tFar >= glm::max(tNear, 0.0f);
}

// Moller-Trumbore, two-sided
__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    const Ray& r)
{
    const glm::vec3 e1 = tri.v1 - tri.v0;
    const glm::vec3 e2 = tri.v2 - tri.v0;
    const glm::vec3 p = glm::cross(r.direction, e2);
    const float det = glm::dot(e1, p);

    if (fabsf(det) < 1e-12f)
    {
        return -1.0f;
    }

    const float invDet = 1.0f / det;
    const glm::vec3 s = r.origin - tri.v0;
    const float u = glm::dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }

    const glm::vec3 q = glm::cross(s, e1);
    const float v = glm::dot(r.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1.0f;
    }

    return glm::dot(e2, q) * invDet;
}

__host__ __device__ float rayBoundsDistance(
    const Ray& r,
    glm::vec3 boundsMin,
    glm::vec3 boundsMax)
{
    const glm::vec3 invDir = 1.0f / r.direction;
    const glm::vec3 t0 = (boundsMin - r.origin) * invDir;
    const glm::vec3 t1 = (boundsMax - r.origin) * invDir;
    const glm::vec3 tSmall = glm::min(t0, t1);
    const glm::vec3 tBig = glm::max(t0, t1);

    const float tNear = glm::max(glm::max(glm::max(tSmall.x, tSmall.y), tSmall.z), 0.0f);
    const float tFar = glm::min(glm::min(tBig.x, tBig.y), tBig.z);
    return tFar >= tNear ? tNear : -1.0f;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    MeshData meshData,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // Not normalized so t matches world space
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f));

    float tMin = FLT_MAX;
    int hitTriangle = -1;

    if (meshData.useBVH)
    {
        int stack[64];
        int stackSize = 0;
        stack[stackSize++] = mesh.bvhRoot;

        while (stackSize > 0)
        {
            const BVHNode node = meshData.nodes[stack[--stackSize]];

            const float tBox = rayBoundsDistance(q, node.boundsMin, node.boundsMax);
            if (tBox < 0.0f || tBox > tMin)
            {
                continue;
            }

            if (node.count > 0)
            {
                for (int i = node.first; i < node.first + node.count; ++i)
                {
                    const float t = triangleIntersectionTest(meshData.triangles[i], q);
                    if (t > 0.0f && t < tMin)
                    {
                        tMin = t;
                        hitTriangle = i;
                    }
                }
            }
            else
            {
                stack[stackSize++] = node.first;
                stack[stackSize++] = node.first + 1;
            }
        }
    }
    else
    {
        if (meshData.useBoundsCulling &&
            !rayHitsBounds(q, mesh.boundsMin, mesh.boundsMax))
        {
            return -1;
        }

        for (int i = mesh.triangleStart; i < mesh.triangleStart + mesh.triangleCount; ++i)
        {
            const float t = triangleIntersectionTest(meshData.triangles[i], q);
            if (t > 0.0f && t < tMin)
            {
                tMin = t;
                hitTriangle = i;
            }
        }
    }

    if (hitTriangle < 0)
    {
        return -1;
    }

    const Triangle& tri = meshData.triangles[hitTriangle];
    const glm::vec3 objectNormal = glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0);

    intersectionPoint = r.origin + tMin * r.direction;
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objectNormal, 0.0f)));
    outside = glm::dot(normal, r.direction) < 0.0f;
    return glm::length(r.origin - intersectionPoint);
}

// Power-8 Mandelbulb distance estimate
__host__ __device__ float mandelbulbDistance(glm::vec3 p)
{
    const float power = 8.0f;
    glm::vec3 z = p;
    float dr = 1.0f;
    float r = glm::length(z);

    for (int i = 0; i < 8 && r < 2.0f; ++i)
    {
        const float theta = acosf(glm::clamp(z.z / r, -1.0f, 1.0f)) * power;
        const float phi = atan2f(z.y, z.x) * power;
        const float zr = powf(r, power);
        dr = powf(r, power - 1.0f) * power * dr + 1.0f;

        z = zr * glm::vec3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)) + p;
        r = glm::length(z);
    }

    return 0.5f * logf(r) * r / dr;
}

// Signed distance to a box centered at the origin: positive outside, negative inside.
__host__ __device__ float boxDistance(glm::vec3 p, glm::vec3 halfSize)
{
    const glm::vec3 overshoot = glm::abs(p) - halfSize;
    const float outside = glm::length(glm::max(overshoot, glm::vec3(0.0f)));
    const float inside = glm::min(glm::max(overshoot.x, glm::max(overshoot.y, overshoot.z)), 0.0f);
    return outside + inside;
}

// Same thing in 2D, for the cross section of a square hole.
__host__ __device__ float squareDistance(float x, float y, float halfSize)
{
    const float dx = fabsf(x) - halfSize;
    const float dy = fabsf(y) - halfSize;
    const float outside = sqrtf(glm::max(dx, 0.0f) * glm::max(dx, 0.0f) + glm::max(dy, 0.0f) * glm::max(dy, 0.0f));
    const float inside = glm::min(glm::max(dx, dy), 0.0f);
    return outside + inside;
}

// Menger sponge in [-1, 1]^3, 4 levels. At each level the cube is split into
// cells, and every cell has a square hole through its middle third along x, y and z.
__host__ __device__ float mengerDistance(glm::vec3 p)
{
    float d = boxDistance(p, glm::vec3(1.0f));
    float cellSize = 2.0f;

    for (int level = 0; level < 4; ++level)
    {
        // Position relative to the center of the cell that p falls in
        const glm::vec3 shifted = p + 1.0f;
        const glm::vec3 local = shifted - cellSize * glm::floor(shifted / cellSize) - 0.5f * cellSize;

        const float holeHalfSize = cellSize / 6.0f;
        const float holeX = squareDistance(local.y, local.z, holeHalfSize);
        const float holeY = squareDistance(local.x, local.z, holeHalfSize);
        const float holeZ = squareDistance(local.x, local.y, holeHalfSize);
        const float hole = glm::min(holeX, glm::min(holeY, holeZ));

        // Carve the holes out of the solid
        d = glm::max(d, -hole);
        cellSize /= 3.0f;
    }

    return d;
}

__host__ __device__ float fractalDistance(GeomType type, glm::vec3 p)
{
    return type == MANDELBULB ? mandelbulbDistance(p) : mengerDistance(p);
}

// Sphere tracing, also works from inside (for glass)
__host__ __device__ float fractalIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    const glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    const glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // Bounding sphere test first
    const float boundRadius = 1.75f;
    const float b = glm::dot(ro, rd);
    const float c = glm::dot(ro, ro) - boundRadius * boundRadius;
    const float disc = b * b - c;
    if (disc < 0.0f)
    {
        return -1;
    }

    const float tFar = -b + sqrtf(disc);
    if (tFar < 0.0f)
    {
        return -1;
    }

    float t = glm::max(-b - sqrtf(disc), 0.0f);
    const float side = fractalDistance(geom.type, ro + t * rd) < 0.0f ? -1.0f : 1.0f;
    const float hitEpsilon = 5e-5f;
    bool hit = false;

    for (int i = 0; i < 256 && t < tFar; ++i)
    {
        const float d = side * fractalDistance(geom.type, ro + t * rd);
        if (d < hitEpsilon)
        {
            hit = true;
            break;
        }
        t += d;
    }

    if (!hit)
    {
        return -1;
    }

    const glm::vec3 p = ro + t * rd;
    const float h = 1e-4f;
    const glm::vec3 gradient(
        fractalDistance(geom.type, p + glm::vec3(h, 0, 0)) - fractalDistance(geom.type, p - glm::vec3(h, 0, 0)),
        fractalDistance(geom.type, p + glm::vec3(0, h, 0)) - fractalDistance(geom.type, p - glm::vec3(0, h, 0)),
        fractalDistance(geom.type, p + glm::vec3(0, 0, h)) - fractalDistance(geom.type, p - glm::vec3(0, 0, h)));

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(p, 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(gradient, 0.0f)));
    outside = side > 0.0f;
    return glm::length(r.origin - intersectionPoint);
}
