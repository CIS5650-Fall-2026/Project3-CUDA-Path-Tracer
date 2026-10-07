#include "intersections.h"
#include "sceneStructs.h"

#define USE_BVH 1
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
        //intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        intersectionPoint = multiplyMV(box.transform, glm::vec4(q.origin + tmin * q.direction, 1.0f));
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

    //glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
    glm::vec3 objspaceIntersection = rt.origin + t * rt.direction;   // was getPointOnRay(rt, t)
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));


    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(Geom tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // ray plane intersection
    float ndotr = glm::dot(r.direction, tri.normal);
    // it is parallel to the plane
    if (glm::abs(ndotr) < EPSILON) return -1.f;

    float t = glm::dot(tri.v1 - r.origin, tri.normal) / ndotr;
    if (t < EPSILON) return -1.f;

    // check if point is within tri bounds
    glm::vec3 pointOnPlane_w = r.origin + t * r.direction;

    // check edge v1 -> v2
    glm::vec3 v1p = pointOnPlane_w - tri.v1;
    glm::vec3 v1v2 = tri.v2 - tri.v1;
    glm::vec3 c = glm::cross(v1v2, v1p);

    if (glm::dot(tri.normal, c) < 0.f) return -1;

    // check edge v2 -> v3
    glm::vec3 v2p = pointOnPlane_w - tri.v2;
    glm::vec3 v2v3 = tri.v3 - tri.v2;
    c = glm::cross(v2v3, v2p);

    if (glm::dot(tri.normal, c) < 0.f) return -1.f;

    // check edge v3-> v1
    glm::vec3 v3p = pointOnPlane_w - tri.v3;
    glm::vec3 v3v1 = tri.v1 - tri.v3;
    c = glm::cross(v3v1, v3p);

    if (glm::dot(tri.normal, c) < 0.f) return -1.f;

    intersectionPoint = pointOnPlane_w;
    normal = tri.normal;
    outside = ndotr < 0;

    return t;
}

__host__ __device__ float bboxIntersectionTest(BVHBounds bbox, Ray r)
{
    // avoid division by zero without skewing the ray
    glm::vec3 d = r.direction;
    d.x = (fabsf(d.x) < 1e-8f) ? copysignf(1e-8f, d.x) : d.x;
    d.y = (fabsf(d.y) < 1e-8f) ? copysignf(1e-8f, d.y) : d.y;
    d.z = (fabsf(d.z) < 1e-8f) ? copysignf(1e-8f, d.z) : d.z;
    glm::vec3 invDir = 1.f / d;

    glm::vec3 nearT = (bbox.minCorner - r.origin) * invDir;
    glm::vec3 farT = (bbox.maxCorner - r.origin) * invDir;

    glm::vec3 tmin = glm::min(nearT, farT);
    glm::vec3 tmax = glm::max(nearT, farT);

    float t0 = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float t1 = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

    // slabs don't overlap, or the whole box is behind the ray
    if (t0 > t1 || t1 <= 0.f)
        return -1.f;

    // entry distance; 0 if the origin is inside the box
    return glm::max(t0, 0.f);
}

__host__ __device__ float bvhNodeIntersectionTest(
    int bvhnodeIdx,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    Geom* geoms,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    int& geomIdx)
{
    float t_min = FLT_MAX;
    geomIdx = -1;
    int stack[64];
    stack[0] = bvhnodeIdx; // push the root node onto the stack
    int stackSize = 1; // currently has size 1

    // we want to add bvhnodes onto our stack when we traverse them
    while (stackSize > 0)
    {
        // pop it off the stack
        stackSize--;
        int currNodeIdx = stack[stackSize];
        BVHNode& node = bvhnodes[currNodeIdx];

        // skip the rest of the loop if we didn't hit anything (test the other node)
        float tbox = bboxIntersectionTest(node.bbox, r);
        if (tbox < 0.f || tbox > t_min) continue;
        
        if (node.isLeaf) // base case
        {
            // itierate through all triangles in the leaf
            for (int i = node.tri_start; i < node.tri_start + node.tri_count; i++)
            {
                glm::vec3 tmp_isect, tmp_nor;
                bool tmp_outside;

                float t = triangleIntersectionTest(geoms[i], r, tmp_isect, tmp_nor, tmp_outside);

                if (t > 0.f && t < t_min)
                {
                    t_min = t;
                    intersectionPoint = tmp_isect;
                    normal = tmp_nor;
                    outside = tmp_outside;
                    geomIdx = i;
                }
            }

        }
        else
        {
            // push the children onto the stack
            stack[stackSize] = node.child_L;
            stackSize++;
            stack[stackSize] = node.child_R;
            stackSize++;
        }
    }
    return t_min;
}


// sub-process of computeIntersections in pathtrace.cu
 __host__ __device__ float geometryIntersectionTest(
    Geom* geoms,
    int geoms_size,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    int& hit_geom_index
)
{

    float t = -1;
    float t_min = FLT_MAX;
    glm::vec3 tmp_intersect(0.f);
    glm::vec3 tmp_normal(0.f);
    bool tmp_outside = false;

    if (USE_BVH && bvhnodes_size > 0)
    {
        // for triangles
        int tmp_geom_index = -1;
        // with t, check for intersection of the ray with the boudning volume
        t = bvhNodeIntersectionTest(0, bvhnodes, bvhnodes_size,
                        geoms, r, tmp_intersect, tmp_normal, tmp_outside, tmp_geom_index);

        if (t > 0.0f && t_min > t) 
        {
                t_min = t;
                hit_geom_index = tmp_geom_index;
                intersectionPoint = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
        }

        // for non triangles, start at very end since the end is all non triangles
        for (int i = geoms_size - 1; i >= 0 && geoms[i].type != TRIANGLE; i--)
        {
            float t = -1.f;
            if (geoms[i].type == CUBE)
                t = boxIntersectionTest(geoms[i], r, tmp_intersect, tmp_normal, tmp_outside);
            else if (geoms[i].type == SPHERE)
                t = sphereIntersectionTest(geoms[i], r, tmp_intersect, tmp_normal, tmp_outside);

            if (t > 0.0f && t < t_min)
            {
                t_min = t;
                hit_geom_index = i;
                intersectionPoint = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }
    } 
    else
    {
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
                t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_outside);
            else if (geom.type == SPHERE)
                t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_outside);
            else if (geom.type == TRIANGLE) 
                t = triangleIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_outside);

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t < t_min)
            {
                t_min = t;
                hit_geom_index = i;
                intersectionPoint = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }
    }
    return hit_geom_index == -1 ? -1.0f : t_min;
}

__host__ __device__ float areaLightIntersectionTest(
    Light light,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float ndotl = glm::dot(light.normal, r.direction);
    // light faces in other direction
    if (ndotl >= 0)
        return -1.f;

    // some point on the light, x local coord transformed onto light
    glm::vec3 P0 = glm::vec3(light.transform * glm::vec4(0.f, 0.f, 0.f, 1.f));
    float t = glm::dot(P0 - r.origin, light.normal) / ndotl;

    if (t < 0.f) return -1.f;

    // check to see if t falls within the bounds of the plane
    glm::vec3 pointOnPlane_w = r.origin + t * r.direction;
    
    // transform to local coords
    glm::vec3 pointOnPlane_l = glm::vec3(light.inverseTransform * glm::vec4(pointOnPlane_w, 1.f));
    if (pointOnPlane_l.x < -0.5 || pointOnPlane_l.x > 0.5 ||
        pointOnPlane_l.z < -0.5 || pointOnPlane_l.z > 0.5)
    {
        // falls outside of bounds of plane
        return -1.f;
    }
    intersectionPoint = pointOnPlane_w;
    normal = light.normal;
    outside = true;
    return t;
}

__host__ __device__ float lightIntersectionTest(
    Light* lights,
    int lights_size,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    int& hit_light_index
)
{
    float t_min = FLT_MAX;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    bool outside = false; // as a placeholder, this is just thrown away
    for (int i = 0; i < lights_size; i++)
    {
        float t = -1;

        Light& light = lights[i];
        if (light.type == AREALIGHT)
        {
            t = areaLightIntersectionTest(light, r, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_light_index = i;
            intersectionPoint = tmp_intersect;
            normal = tmp_normal;
        }
    }
    return t_min;
}