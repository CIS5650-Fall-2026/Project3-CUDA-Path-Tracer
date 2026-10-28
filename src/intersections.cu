#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    glm::vec2 &uv,
    glm::vec3 &tangent,
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

        // uv from the object-space hit point, where the cube spans -0.5 to 0.5: drop the
        // axis the face normal points along, the other two shifted by 0.5 cover [0, 1]
        // Each face is oriented as seen from outside, with v = 0 at the top of the image
        glm::vec3 p = getPointOnRay(q, tmin);
        glm::vec3 faceN = outside ? tmin_n : -tmin_n;
        uv = glm::vec2(0.0f);
        // `right` is the object-space direction in which u grows on this face
        glm::vec3 right(1.0f, 0.0f, 0.0f);
        if (faceN.z > 0.5f)
        {
            // +z: x runs right and y runs up, so v counts down from the top
            uv = glm::vec2(p.x + 0.5f, 0.5f - p.y);
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        else if (faceN.z < -0.5f)
        {
            // -z: seen from behind, x runs left
            uv = glm::vec2(0.5f - p.x, 0.5f - p.y);
            right = glm::vec3(-1.0f, 0.0f, 0.0f);
        }
        else if (faceN.x > 0.5f)
        {
            // +x: z runs left, away from a viewer standing on the +x side
            uv = glm::vec2(0.5f - p.z, 0.5f - p.y);
            right = glm::vec3(0.0f, 0.0f, -1.0f);
        }
        else if (faceN.x < -0.5f)
        {
            // -x: z runs right
            uv = glm::vec2(p.z + 0.5f, 0.5f - p.y);
            right = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        else if (faceN.y > 0.5f)
        {
            // +y: seen from above with x to the right, the top edge is the -z side
            uv = glm::vec2(p.x + 0.5f, p.z + 0.5f);
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        else
        {
            // -y: seen from below with x to the right, the top edge is the +z side
            uv = glm::vec2(p.x + 0.5f, 0.5f - p.z);
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        // a direction, so w = 0 and the plain transform, not the inverse transpose
        tangent = glm::normalize(multiplyMV(box.transform, glm::vec4(right, 0.0f)));
        // inside hits report the outward normal, same convention as sphere and mesh
        if (!outside) normal = -normal;
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


__host__ __device__ bool aabbIntersectionTest(
    glm::vec3 aabbMin,
    glm::vec3 aabbMax,
    Ray r)
{
    // slab test; a zero direction component gives +-inf, which min/max handle fine
    glm::vec3 invDir = 1.0f / r.direction;
    glm::vec3 t0 = (aabbMin - r.origin) * invDir;
    glm::vec3 t1 = (aabbMax - r.origin) * invDir;
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar = glm::max(t0, t1);

    // enter = latest slab entry, exit = earliest slab exit
    float tEnter = glm::max(glm::max(tNear.x, tNear.y), tNear.z);
    float tExit = glm::min(glm::min(tFar.x, tFar.y), tFar.z);
    // tExit > 0 (not tEnter) so a ray starting inside the box still counts
    return tEnter <= tExit && tExit > 0.0f;
}

// Ray/triangle test without back-face culling, so rays leaving
// a closed mesh (refraction) still hit its inside. Same math as
// glm::intersectRayTriangle, which rejects a negative determinant
// bary.x/.y are the weights of vert1/vert2 (vert0 gets 1 - x - y), bary.z is t
__host__ __device__ bool rayTriangleNoCull(
    const glm::vec3& orig,
    const glm::vec3& dir,
    const glm::vec3& vert0,
    const glm::vec3& vert1,
    const glm::vec3& vert2,
    glm::vec3& bary)
{
    // below this the ray runs parallel to the triangle
    const float parallelEps = 1e-7f;

    glm::vec3 e1 = vert1 - vert0;
    glm::vec3 e2 = vert2 - vert0;

    // determinant of the (t, u, v) system; its sign only says which side the ray
    // came from, so reject just the near-zero (parallel) case
    glm::vec3 p = cross(dir, e2);
    float a = dot(e1, p);
    if (fabs(a) < parallelEps) return false;

    float f = 1.0f / a;
    glm::vec3 s = orig - vert0;
    bary.x = f * dot(s, p);
    if (bary.x < 0 || bary.x > 1) return false;

    glm::vec3 q = cross(s, e1);
    bary.y = f * dot(dir, q);
    if (bary.y < 0 || bary.x + bary.y > 1) return false;

    bary.z = f * dot(e2, q);
    return bary.z >= 0;
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    bool& outside)
{
    // below this the triangle's uvs have no area and cannot define a tangent
    const float degenerateUV = 1e-8f;

    glm::vec3 bary;
    bool hit = rayTriangleNoCull(r.origin, r.direction, triangle.vertices[0], triangle.vertices[1], triangle.vertices[2], bary);
    // bary is unwritten on a miss, so check before reading it
    if (!hit || bary.z <= 0.0f) return -1.0f;

    float t = bary.z;
    intersectionPoint = getPointOnRay(r, t);

    // smooth shading: interpolate vertex normals with the barycentric weights
    float w0 = 1 - bary.x - bary.y;
    float w1 = bary.x;
    float w2 = bary.y;
    normal = glm::normalize(w0 * triangle.normals[0] + w1 * triangle.normals[1] + w2 * triangle.normals[2]);
    uv = w0 * triangle.uvs[0] + w1 * triangle.uvs[1] + w2 * triangle.uvs[2];

    // tangent: the direction along the triangle in which u grows
    // Along edge1 the position changes by e1 while uv changes by duv1, same for edge2:
    //     e1 = T * duv1.x + B * duv1.y
    //     e2 = T * duv2.x + B * duv2.y
    // two equations, two unknown vectors T and B; solving for T gives
    //     T = (e1 * duv2.y - e2 * duv1.y) / (duv1.x * duv2.y - duv2.x * duv1.y)
    glm::vec3 e1 = triangle.vertices[1] - triangle.vertices[0];
    glm::vec3 e2 = triangle.vertices[2] - triangle.vertices[0];
    glm::vec2 duv1 = triangle.uvs[1] - triangle.uvs[0];
    glm::vec2 duv2 = triangle.uvs[2] - triangle.uvs[0];
    // the sign of denom flips for mirrored uvs, so the division has to stay
    float denom = duv1.x * duv2.y - duv2.x * duv1.y;
    if (fabs(denom) < degenerateUV)
        tangent = glm::normalize(e1);
    else
        tangent = glm::normalize((e1 * duv2.y - e2 * duv1.y) / denom);

    outside = dot(r.direction, normal) < 0;
    return t;
}
