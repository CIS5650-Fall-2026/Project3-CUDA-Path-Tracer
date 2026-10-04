#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal)
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
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
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
    glm::vec3 &normal)
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
    }
    else
    {
        t = max(t1, t2);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float anyGeomIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal) {

    if (geom.type == CUBE)
    {
        return boxIntersectionTest(geom, r, intersectionPoint, normal);
    }
    else if (geom.type == SPHERE)
    {
        return sphereIntersectionTest(geom, r, intersectionPoint, normal);
    }
    else if (geom.type == TRIANGLE)
    {
        return triangleGeomIntersectionTest(geom, r, intersectionPoint, normal);
    }

	// undefined geometry type: return -1.
    return -1.0f;

}




// Triangle Intersection, Möller–Trumbore intersection algorithm
__host__ __device__ float triangleGeomIntersectionTest(
    Geom triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
    glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

	glm::vec3 v0 = triangle.vertices[0];
	glm::vec3 v1 = triangle.vertices[1];
	glm::vec3 v2 = triangle.vertices[2];

	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;

	glm::vec3 nor = glm::normalize(glm::cross(e1, e2));

	glm::vec3 h = glm::cross(rt.direction, e2);
	float a = glm::dot(e1, h);

    if (fabs(a) < 1e-8)
		return -1.0f; // if a close to 0 - ray is parallel and no intersection

	float f = 1.0f / a;

	// u and v are bary. coordinates of the intersection point w/ respect to the triangle
	glm::vec3 s = rt.origin - v0;
	float u = f * glm::dot(s, h);
	glm::vec3 q = glm::cross(s, e1);
	float v = f * glm::dot(rt.direction, q);

    if (u < 0.0f || u > 1.0f || v < 0.0f || u + v > 1.0f)
		return -1.0f; // no intersection

	float t = f * glm::dot(e2, q);
    if (t < 1e-8) return -1.0f; // intersection behind ray origin
		
	// return successful intersection
    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(getPointOnRay(rt, t), 1.f));
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(nor, 0.f)));

    return glm::length(r.origin - intersectionPoint);

}

__host__ __device__ bool AABBIntersectionTest(
    const BoundingBox& bbox,
    const Ray& ray,
    float& tMin,
    float& tMax)
{
    tMin = 0.0f;
    tMax = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis)
    {
        float invD = 1.0f / ray.direction[axis];

        float t0 = (bbox.min[axis] - ray.origin[axis]) * invD;
        float t1 = (bbox.max[axis] - ray.origin[axis]) * invD;

        if (invD < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tMin = glm::max(tMin, t0);
        tMax = glm::min(tMax, t1);

        if (tMax < tMin)
            return false;
    }

    return true;
}

// 
__host__ __device__
float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& ray,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
    glm::vec3 e1 = triangle.v1 - triangle.v0;
    glm::vec3 e2 = triangle.v2 - triangle.v0;

    normal = glm::normalize(glm::cross(e1, e2));

    glm::vec3 h = glm::cross(ray.direction, e2);
    float a = glm::dot(e1, h);

    if (fabs(a) < 1e-8f)
        return -1.0f;

    float f = 1.0f / a;

    glm::vec3 s = ray.origin - triangle.v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;

    float t = f * glm::dot(e2, q);

    if (t < 1e-8f)
        return -1.0f;

    intersectionPoint = getPointOnRay(ray, t);

    return t;
}


__device__ bool BVHIntersectionTest(
    int rootIndex,
    const Ray& ray,
    BVHNode* bvhNodes,
    Triangle* triangles,
    Geom* geoms,
    float& closestT,
    glm::vec3& hitPoint,
    glm::vec3& hitNormal,
    int& materialId)
{
    int stack[BVH_DEPTH];
    int stackSize = 0;

    stack[stackSize++] = rootIndex;

    bool hit = false;

    while (stackSize > 0)
    {
        int nodeIndex = stack[--stackSize];

        const BVHNode& node = bvhNodes[nodeIndex];

        float tMin, tMax;

        if (!AABBIntersectionTest(node.bbox, ray, tMin, tMax))
            continue;

        if (node.isLeaf)
        {
            if (node.triangleIndex != -1)
            {
                const Triangle& tri =
                    triangles[node.triangleIndex];

                glm::vec3 point;
                glm::vec3 normal;

                float t = triangleIntersectionTest(
                    tri, ray, point, normal);

                if (t > 0.0f && t < closestT)
                {
                    closestT = t;
                    hitPoint = point;
                    hitNormal = normal;
                    materialId = node.materialId;
                    hit = true;
                }
            }
            else if (node.geomIndex != -1)
            {
                const Geom& geom =
                    geoms[node.geomIndex];

                glm::vec3 point;
                glm::vec3 normal;

                float t = anyGeomIntersectionTest(
                    geom, ray, point, normal);

                if (t > 0.0f && t < closestT)
                {
                    closestT = t;
                    hitPoint = point;
                    hitNormal = normal;
                    materialId = node.materialId;
                    hit = true;
                }
            }

            continue;
        }

        if (node.leftChild != -1)
            stack[stackSize++] = node.leftChild;

        if (node.rightChild != -1)
            stack[stackSize++] = node.rightChild;
    }

    return hit;

}