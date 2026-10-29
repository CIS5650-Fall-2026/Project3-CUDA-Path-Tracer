// OptiX device programs, compiled to OptiX-IR; path-ray hits match what computeIntersections writes.

#include <optix.h>

#include "render/mesh_hit.h"
#include "optix/optix_params.h"

extern "C"
{
    __constant__ OptixIntersectParams params;
}

static __forceinline__ __device__ float3 toFloat3(glm::vec3 v)
{
    return make_float3(v.x, v.y, v.z);
}

static __forceinline__ __device__ glm::vec3 toVec3(float3 v)
{
    return glm::vec3(v.x, v.y, v.z);
}

// One thread per path, then one per queued shadow ray; path directions are normalized so t is a world distance.
extern "C" __global__ void __raygen__paths()
{
    const unsigned int i = optixGetLaunchIndex().x;
    if (i < (unsigned int)params.numPaths)
    {
        // Ended paths stay in place when compaction is off
        if (params.paths[i].remainingBounces <= 0)
        {
            return;
        }
        const Ray ray = params.paths[i].ray;
        optixTrace(params.handle,
            toFloat3(ray.origin),
            toFloat3(glm::normalize(ray.direction)),
            0.0f,                  // tmin: the naive tests accept any t > 0 too
            1e16f,
            0.0f,                  // ray time
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,                     // SBT offset
            1,                     // SBT stride: one set of hit groups for both kinds of ray
            0);                    // miss program index
        return;
    }
    const unsigned int s = i - params.numPaths;
    if (s >= (unsigned int)*params.shadowCount)
    {
        return;
    }
    const ShadowRay& ray = params.shadowRays[s];
    optixTrace(params.handle,
        toFloat3(ray.origin),
        toFloat3(ray.direction),
        0.0f,                      // the origin already sits off the surface
        ray.tMax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0,
        1,
        1);                        // __miss__shadow
}

extern "C" __global__ void __miss__paths()
{
    const unsigned int i = optixGetLaunchIndex().x;
    params.intersections[i].t = -1.0f;
    params.materialIds[i] = params.numMaterials;
}

// No atomic: a pixel has one path, so at most one shadow ray per launch adds to it.
extern "C" __global__ void __miss__shadow()
{
    const ShadowRay& ray = params.shadowRays[optixGetLaunchIndex().x - params.numPaths];
    params.image[ray.pixelIndex] += ray.contribution;
}

static __forceinline__ __device__ InstanceRecord instance()
{
    return params.instances[optixGetInstanceId()];
}

// geometricNormal is unit length in world space.
static __forceinline__ __device__ void writeHit(const InstanceRecord& inst, glm::vec3 normal, glm::vec2 uv,
    glm::vec4 tangent, bool outside, glm::vec3 geometricNormal)
{
    const unsigned int i = optixGetLaunchIndex().x;

    ShadeableIntersection isect;
    isect.t = optixGetRayTmax();
    isect.surfaceNormal = normal;
    isect.uv = uv;
    isect.tangent = tangent;
    isect.materialId = inst.materialId;
    isect.outside = outside;
    isect.cosGeometric = fabsf(glm::dot(geometricNormal, toVec3(optixGetWorldRayDirection())));
    isect.lightMask = inst.lightMask;
    isect.lightGroup = inst.lightGroup;
    params.intersections[i] = isect;
    params.materialIds[i] = inst.materialId;
}

// Like boxIntersectionTest, the normal faces the ray: inward on the exit hit from inside.
extern "C" __global__ void __closesthit__cube()
{
    const unsigned int face = optixGetPrimitiveIndex() / 2;
    float n[3] = { 0.0f, 0.0f, 0.0f };
    n[face / 2] = (face & 1) ? 1.0f : -1.0f;

    const float3 worldNormal = optixTransformNormalFromObjectToWorldSpace(make_float3(n[0], n[1], n[2]));
    // A closest-hit program cannot read the object-space ray, so the world ray is transformed by hand.
    const float3 origin = optixTransformPointFromWorldToObjectSpace(optixGetWorldRayOrigin());
    const bool outside = fmaxf(fabsf(origin.x), fmaxf(fabsf(origin.y), fabsf(origin.z))) > 0.5f;

    glm::vec3 normal = glm::normalize(toVec3(worldNormal));
    if (!outside)
    {
        normal = -normal;
    }
    writeHit(instance(), normal, glm::vec2(0.0f), glm::vec4(0.0f), outside, normal);
}

// Radius 0.5 at the origin; the built-in sphere never hits from inside (OptiX Programming Guide 9.1, "Curves and Spheres").
// The object-space direction stays unnormalized, so t is the same world distance in both spaces.
extern "C" __global__ void __intersection__sphere()
{
    const glm::vec3 o = toVec3(optixGetObjectRayOrigin());
    const glm::vec3 d = toVec3(optixGetObjectRayDirection());

    // |o + t d|^2 = r^2  ->  a t^2 + 2 b t + c = 0
    const float a = glm::dot(d, d);
    const float b = glm::dot(o, d);
    const float c = glm::dot(o, o) - 0.25f;
    const float discriminant = b * b - a * c;
    if (discriminant < 0.0f)
    {
        return;
    }
    const float root = sqrtf(discriminant);
    const float tNear = (-b - root) / a;
    const float tFar = (-b + root) / a;

    float t;
    bool outside;
    if (tNear > 0.0f && tFar > 0.0f)
    {
        t = tNear;
        outside = true;
    }
    else if (tFar > 0.0f)
    {
        t = tFar;
        outside = false;
    }
    else
    {
        return;
    }

    // Attributes: the object-space hit point, which is also the normal direction, then outside.
    const glm::vec3 hit = o + t * d;
    optixReportIntersection(t, 0,
        __float_as_uint(hit.x), __float_as_uint(hit.y), __float_as_uint(hit.z),
        static_cast<unsigned int>(outside));
}

extern "C" __global__ void __closesthit__sphere()
{
    const float3 hit = make_float3(__uint_as_float(optixGetAttribute_0()),
                                   __uint_as_float(optixGetAttribute_1()),
                                   __uint_as_float(optixGetAttribute_2()));
    const bool outside = optixGetAttribute_3() != 0;

    glm::vec3 normal = glm::normalize(toVec3(optixTransformNormalFromObjectToWorldSpace(hit)));
    if (!outside)
    {
        normal = -normal;
    }

    writeHit(instance(), normal, glm::vec2(0.0f), glm::vec4(0.0f), outside, normal);
}

// Same conventions as meshIntersectionTest: the geometric normal decides outside.
extern "C" __global__ void __closesthit__mesh()
{
    const InstanceRecord inst = instance();
    const TriangleMesh mesh = params.buffers.meshes[inst.meshId];
    const glm::ivec3 tri = params.buffers.indices[mesh.indexOffset + optixGetPrimitiveIndex()];
    const glm::vec3 p0 = params.buffers.positions[tri.x];
    const glm::vec3 p1 = params.buffers.positions[tri.y];
    const glm::vec3 p2 = params.buffers.positions[tri.z];

    // Only the sign against the geometric normal matters, which object and world space share.
    const glm::vec3 d = toVec3(optixTransformVectorFromWorldToObjectSpace(optixGetWorldRayDirection()));
    const glm::vec3 geometricNormal = glm::cross(p1 - p0, p2 - p0);
    const bool outside = glm::dot(geometricNormal, d) < 0.0f;

    const auto normalToWorld = [](glm::vec3 n) { return toVec3(optixTransformNormalFromObjectToWorldSpace(toFloat3(n))); };
    const auto vectorToWorld = [](glm::vec3 t) { return toVec3(optixTransformVectorFromObjectToWorldSpace(toFloat3(t))); };
    const float2 bary = optixGetTriangleBarycentrics();
    const glm::vec3 normal = meshShadingNormal(interpolate(params.buffers.normals, tri, bary.x, bary.y), geometricNormal,
        outside, toVec3(optixGetWorldRayDirection()), normalToWorld);
    const glm::vec2 uv = interpolate(params.buffers.uvs, tri, bary.x, bary.y);
    const glm::vec4 tangent = meshTangent(params.buffers.tangents, tri, bary.x, bary.y, inst.tangentSign, vectorToWorld);

    writeHit(inst, normal, uv, tangent, outside, glm::normalize(normalToWorld(geometricNormal)));
}

// OptiX may call this more than once per triangle, so the cut-out test must give the same answer each time.
extern "C" __global__ void __anyhit__mesh()
{
    const InstanceRecord inst = instance();
    const TriangleMesh mesh = params.buffers.meshes[inst.meshId];
    const unsigned int triangle = optixGetPrimitiveIndex();
    const glm::ivec3 tri = params.buffers.indices[mesh.indexOffset + triangle];
    const float2 bary = optixGetTriangleBarycentrics();
    const glm::vec2 uv = interpolate(params.buffers.uvs, tri, bary.x, bary.y);
    const unsigned int i = optixGetLaunchIndex().x;
    unsigned int alphaSeed;
    if (i < (unsigned int)params.numPaths)
    {
        const PathSegment& path = params.paths[i];
        alphaSeed = alphaPathSeed(params.iter, path.pixelIndex, path.remainingBounces);
    }
    else
    {
        alphaSeed = params.shadowRays[i - params.numPaths].alphaSeed;
    }
    if (alphaCutOut(params.materials[inst.materialId], uv, params.textures, alphaSeed, optixGetInstanceId(), triangle))
    {
        optixIgnoreIntersection();
    }
}
