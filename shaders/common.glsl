#ifndef COMMON_GLSL
#define COMMON_GLSL

// This must always come first, since it includes extensions and structs
// shared between Cpp and GLSL
#include "gpu/shared.h"

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBuffer
{
    Vertex vertices[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer IndexBuffer
{
    uint indices[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer GeometryBuffer
{
    GeometryInfo geometries[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MaterialBuffer
{
    GpuMaterial materials[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer LightBuffer
{
    EmissiveTri lights[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer SceneAddressesBuffer
{
    SceneAddresses deviceAddresses;
};

// Constants/Sentinels
const float kMiss = -1.0;
const float kTMin = 0.001;
const float kTMax = 10000.0;
const float kShadowRayEpsilon = 0.00001;
const GpuMaterial kEmptyMaterial = GpuMaterial(
    vec3(0.0),
    0u,
    vec3(0.0),
    0.0,
    0.0,
    0.0,
    0.0);

struct IntersectionInfo
{
    // World-space position
    vec3 worldPos;

    // The interpolated world-space normal, used for shading computations
    vec3 worldShadingNor;

    // The cross product of world-space triangle edges, guaranteed
    // to be outward-facing. This vector is ray-independent.
    vec3 worldGeometricNor;

    vec2 uv;
    uint materialIndex;
    
    // kMiss (-1.0) indicates a miss
    float t;
};

struct SurfaceInteraction
{
    IntersectionInfo isect;
    GpuMaterial material;
    vec3 faceForwardGeometricNor;
};

// hitAttributeEXT gives the weights for two vertices,
// which we use to compute the third
vec3 getBarycentricWeights(vec2 attribs)
{
    return vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec2 interpolateVec2(vec3 barycentricWeights, vec2 attr0, vec2 attr1, vec2 attr2)
{
    return barycentricWeights.x * attr0
        + barycentricWeights.y * attr1
        + barycentricWeights.z * attr2;
}

vec3 interpolateVec3(vec3 barycentricWeights, vec3 attr0, vec3 attr1, vec3 attr2)
{
    return barycentricWeights.x * attr0
        + barycentricWeights.y * attr1
        + barycentricWeights.z * attr2;
}

IntersectionInfo getIntersectionInfo(
    uint64_t sceneAddr,
    uint instanceFirstGeometry,
    uint localBLASGeometryIndex,
    uint primitiveId,
    vec3 barycentricWeights,
    mat3 worldToObject,
    vec3 rayOrigin,
    vec3 rayDir,
    float t)
{
    SceneAddresses sceneAddresses = SceneAddressesBuffer(sceneAddr).deviceAddresses;

    uint globalGeometryIndex = instanceFirstGeometry + localBLASGeometryIndex;
    GeometryInfo geomInfo = GeometryBuffer(sceneAddresses.geometriesAddr).geometries[globalGeometryIndex];

    // Note: our "global" indices buffer holds values that are actually primitive-local
    uint indicesBaseIdx = geomInfo.indexOffset + primitiveId * 3;
    uint primLocalIdx0 = IndexBuffer(sceneAddresses.indicesAddr).indices[indicesBaseIdx];
    uint primLocalIdx1 = IndexBuffer(sceneAddresses.indicesAddr).indices[indicesBaseIdx + 1];
    uint primLocalIdx2 = IndexBuffer(sceneAddresses.indicesAddr).indices[indicesBaseIdx + 2];

    Vertex v0 = VertexBuffer(sceneAddresses.verticesAddr).vertices[geomInfo.vertexOffset + primLocalIdx0];
    Vertex v1 = VertexBuffer(sceneAddresses.verticesAddr).vertices[geomInfo.vertexOffset + primLocalIdx1];
    Vertex v2 = VertexBuffer(sceneAddresses.verticesAddr).vertices[geomInfo.vertexOffset + primLocalIdx2];

    // Geometric normal calculation assumes CCW winding (standard for glTF)
    vec3 modelGeometricNor = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    vec3 modelShadingNor = normalize(interpolateVec3(barycentricWeights, v0.normal, v1.normal, v2.normal));
    vec2 uv = interpolateVec2(barycentricWeights, v0.uv, v1.uv, v2.uv);

    // This is equivalent to transpose(inverse(modelMatrix))
    mat3 normalMatrix = transpose(worldToObject);

    vec3 worldGeometricNor = normalize(normalMatrix * modelGeometricNor);

    // If the determinant of worldToObject/objectToWorld is negative, it means
    // that vertex winding flips after application. In this case we need to negate
    // the geometric normal so it accurately matches the winding.
    // By guaranteeing that the geometric normal matches winding, we can safely
    // use dot(rayDir, geometricNormal) to tell us which side of a surface we
    // have hit (this matters for things like refraction, where we can intersect
    // a surface from both inside and outside the object)
    if (determinant(worldToObject) < 0.0)
    {
        worldGeometricNor = -worldGeometricNor;
    }

    vec3 worldShadingNor = normalize(normalMatrix * modelShadingNor);

    IntersectionInfo isect;
    isect.worldPos = rayOrigin + rayDir * t;
    isect.worldShadingNor = worldShadingNor;
    isect.worldGeometricNor = worldGeometricNor;
    isect.uv = uv;
    isect.materialIndex = geomInfo.materialIndex;
    isect.t = t;

    return isect;
}

#endif