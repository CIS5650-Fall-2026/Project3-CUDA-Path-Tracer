#include "intersections.h"

#include "utilities.h"

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

__host__ __device__ float mandelbulbDistance(glm::vec3 p)
{
    p *= 2.4f;
    glm::vec3 z = p;
    float dr = 1.0f;
    float r = glm::length(z);
    for (int i = 0; i < 8 && r < 2.0f; i++)
    {
        float theta = acos(z.z / r) * 8.0f;
        float phi = atan2(z.y, z.x) * 8.0f;
        dr = powf(r, 7.0f) * 8.0f * dr + 1.0f;
        z = powf(r, 8.0f) * glm::vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + p;
        r = glm::length(z);
    }
    return 0.5f * log(r) * r / dr / 2.4f;
}

__host__ __device__ float mengerDistance(glm::vec3 p)
{
    p *= 2.0f;
    glm::vec3 q = glm::abs(p) - 1.0f;
    float d = glm::length(glm::max(q, 0.0f)) + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
    float scale = 1.0f;
    for (int i = 0; i < 4; i++)
    {
        glm::vec3 a = glm::mod(p * scale, 2.0f) - 1.0f;
        scale *= 3.0f;
        glm::vec3 r = glm::abs(1.0f - 3.0f * glm::abs(a));
        float cross = glm::min(glm::max(r.x, r.y), glm::min(glm::max(r.y, r.z), glm::max(r.z, r.x)));
        d = glm::max(d, (cross - 1.0f) / scale);
    }
    return d / 2.0f;
}

__host__ __device__ float shapeDistance(GeomType type, glm::vec3 p)
{
    return type == MANDELBULB ? mandelbulbDistance(p) : mengerDistance(p);
}

__host__ __device__ float distanceFieldIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 t1 = (glm::vec3(-0.5f) - ro) / rd;
    glm::vec3 t2 = (glm::vec3(0.5f) - ro) / rd;
    glm::vec3 tNear = glm::min(t1, t2);
    glm::vec3 tFar = glm::max(t1, t2);
    float t = glm::max(glm::max(tNear.x, glm::max(tNear.y, tNear.z)), 0.0f);
    float tEnd = glm::min(tFar.x, glm::min(tFar.y, tFar.z));

    const float epsilon = 0.0002f;
    for (int i = 0; i < 128 && t < tEnd; i++)
    {
        glm::vec3 p = ro + rd * t;
        float distance = shapeDistance(geom.type, p);
        if (distance < epsilon)
        {
            glm::vec3 dx(epsilon, 0.0f, 0.0f);
            glm::vec3 dy(0.0f, epsilon, 0.0f);
            glm::vec3 dz(0.0f, 0.0f, epsilon);
            glm::vec3 gradient(
                shapeDistance(geom.type, p + dx) - shapeDistance(geom.type, p - dx),
                shapeDistance(geom.type, p + dy) - shapeDistance(geom.type, p - dy),
                shapeDistance(geom.type, p + dz) - shapeDistance(geom.type, p - dz));

            outside = true;
            intersectionPoint = multiplyMV(geom.transform, glm::vec4(p, 1.0f));
            normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(gradient, 0.0f)));
            return glm::length(r.origin - intersectionPoint);
        }
        t += distance;
    }

    return -1;
}

__host__ __device__ bool boundsIntersectionTest(glm::vec3 boundsMin, glm::vec3 boundsMax, Ray r, float maxDistance)
{
    glm::vec3 t1 = (boundsMin - r.origin) / r.direction;
    glm::vec3 t2 = (boundsMax - r.origin) / r.direction;
    glm::vec3 tNear = glm::min(t1, t2);
    glm::vec3 tFar = glm::max(t1, t2);
    float tEnter = glm::max(tNear.x, glm::max(tNear.y, tNear.z));
    float tExit = glm::min(tFar.x, glm::min(tFar.y, tFar.z));
    return tExit >= glm::max(tEnter, 0.0f) && tEnter < maxDistance;
}

__host__ __device__ float triangleIntersectionTest(const Triangle &triangle, Ray r, glm::vec3 &bary)
{
    if (glm::intersectRayTriangle(r.origin, r.direction,
        triangle.positions[0], triangle.positions[1], triangle.positions[2], bary))
    {
        return bary.z;
    }
    if (glm::intersectRayTriangle(r.origin, r.direction,
        triangle.positions[0], triangle.positions[2], triangle.positions[1], bary))
    {
        bary = glm::vec3(bary.y, bary.x, bary.z);
        return bary.z;
    }
    return -1;
}

__host__ __device__ void closestTriangle(
    const Triangle *triangles,
    int start,
    int count,
    Ray r,
    float &tMin,
    int &hitTriangle,
    glm::vec3 &hitBary)
{
    for (int i = start; i < start + count; i++)
    {
        glm::vec3 bary;
        float t = triangleIntersectionTest(triangles[i], r, bary);
        if (t > 0.0f && t < tMin)
        {
            tMin = t;
            hitTriangle = i;
            hitBary = bary;
        }
    }
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    const Triangle *triangles,
    const BVHNode *nodes,
    bool useBvh,
    bool cullBounds,
    ShadeableIntersection &hit)
{
    float tMin = FLT_MAX;
    int hitTriangle = -1;
    glm::vec3 bary;
    if (useBvh)
    {
        int stack[64];
        int stackSize = 0;
        stack[stackSize++] = mesh.bvhRoot;
        while (stackSize > 0)
        {
            const BVHNode &node = nodes[stack[--stackSize]];
            if (!boundsIntersectionTest(node.boundsMin, node.boundsMax, r, tMin))
            {
                continue;
            }
            if (node.triangleCount > 0)
            {
                closestTriangle(triangles, node.triangleStart, node.triangleCount, r, tMin, hitTriangle, bary);
            }
            else
            {
                stack[stackSize++] = node.left;
                stack[stackSize++] = node.right;
            }
        }
    }
    else if (!cullBounds || boundsIntersectionTest(mesh.boundsMin, mesh.boundsMax, r, tMin))
    {
        closestTriangle(triangles, mesh.triangleStart, mesh.triangleCount, r, tMin, hitTriangle, bary);
    }
    if (hitTriangle < 0)
    {
        return -1;
    }

    const Triangle &triangle = triangles[hitTriangle];
    float w = 1.0f - bary.x - bary.y;
    glm::vec3 edge1 = triangle.positions[1] - triangle.positions[0];
    glm::vec3 edge2 = triangle.positions[2] - triangle.positions[0];
    glm::vec2 uvEdge1 = triangle.uvs[1] - triangle.uvs[0];
    glm::vec2 uvEdge2 = triangle.uvs[2] - triangle.uvs[0];
    float determinant = uvEdge1.x * uvEdge2.y - uvEdge2.x * uvEdge1.y;

    hit.materialId = triangle.materialId;
    hit.uv = w * triangle.uvs[0] + bary.x * triangle.uvs[1] + bary.y * triangle.uvs[2];
    hit.tangent = determinant != 0.0f ? (edge1 * uvEdge2.y - edge2 * uvEdge1.y) / determinant : edge1;
    hit.surfaceNormal = glm::normalize(
        w * triangle.normals[0] + bary.x * triangle.normals[1] + bary.y * triangle.normals[2]);
    hit.outside = glm::dot(glm::cross(edge1, edge2), r.direction) < 0.0f;
    if (!hit.outside)
    {
        hit.surfaceNormal = -hit.surfaceNormal;
    }
    return tMin;
}

__host__ __device__ void primitiveUV(const Geom &geom, glm::vec3 point, glm::vec2 &uv, glm::vec3 &tangent)
{
    glm::vec3 p = multiplyMV(geom.inverseTransform, glm::vec4(point, 1.0f));
    glm::vec3 objectTangent;
    glm::vec3 a = glm::abs(p);
    if (geom.type == SPHERE)
    {
        uv = glm::vec2(0.5f - atan2(p.z, p.x) / TWO_PI, acos(glm::clamp(p.y * 2.0f, -1.0f, 1.0f)) / PI);
        objectTangent = glm::vec3(p.z, 0.0f, -p.x);
    }
    else if (a.x >= a.y && a.x >= a.z)
    {
        float side = p.x > 0.0f ? 1.0f : -1.0f;
        uv = glm::vec2(0.5f - side * p.z, 0.5f - p.y);
        objectTangent = glm::vec3(0.0f, 0.0f, -side);
    }
    else if (a.y >= a.z)
    {
        float side = p.y > 0.0f ? 1.0f : -1.0f;
        uv = glm::vec2(0.5f + p.x, 0.5f + side * p.z);
        objectTangent = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        float side = p.z > 0.0f ? 1.0f : -1.0f;
        uv = glm::vec2(0.5f + side * p.x, 0.5f - p.y);
        objectTangent = glm::vec3(side, 0.0f, 0.0f);
    }
    tangent = multiplyMV(geom.transform, glm::vec4(objectTangent, 0.0f));
}
