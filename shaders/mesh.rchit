#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "gpu/shared.h"
#include "common.glsl"

// This shader will fill the payload, which is then read by raygen after calling traceRayEXT.
// The location must match the payload location raygen passes to traceRayEXT.
layout(location = 0) rayPayloadInEXT IntersectionInfo payload;

// This is written by the HW RT's built-in triangle intersection.
// It contains barycentric weights for vertices 1 and 2.
hitAttributeEXT vec2 attribs;

layout(push_constant, scalar) uniform PushConstantBlock
{
    PushConstants pc;
};

void main()
{
    payload = getIntersectionInfo(
        pc.sceneAddr,
        gl_InstanceCustomIndexEXT,
        gl_GeometryIndexEXT,
        gl_PrimitiveID,
        getBarycentricWeights(attribs),
        mat3(gl_WorldToObjectEXT),
        gl_WorldRayOriginEXT,
        gl_WorldRayDirectionEXT,
        gl_HitTEXT);
}
