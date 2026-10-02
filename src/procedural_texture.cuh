#pragma once

#include "glm/glm.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define PROCEDURAL_TEXTURE_HD __host__ __device__
#define PROCEDURAL_TEXTURE_INLINE __forceinline__
#else
#define PROCEDURAL_TEXTURE_HD
#define PROCEDURAL_TEXTURE_INLINE inline
#endif

namespace renderer
{
namespace procedural_texture_detail
{

constexpr float kTwoPi = 6.28318530717958647692f;

PROCEDURAL_TEXTURE_HD PROCEDURAL_TEXTURE_INLINE float fract(float value)
{
    return value - floorf(value);
}

PROCEDURAL_TEXTURE_HD PROCEDURAL_TEXTURE_INLINE float clampUnit(float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

PROCEDURAL_TEXTURE_HD PROCEDURAL_TEXTURE_INLINE float smoothstep(
    float edge0, float edge1, float value)
{
    const float t = clampUnit((value - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

PROCEDURAL_TEXTURE_HD PROCEDURAL_TEXTURE_INLINE glm::vec3 mix(
    const glm::vec3& first, const glm::vec3& second, float weight)
{
    return first * (1.0f - weight) + second * weight;
}

} // namespace procedural_texture_detail

PROCEDURAL_TEXTURE_HD PROCEDURAL_TEXTURE_INLINE glm::vec3
evaluateProceduralColor(int id, const glm::vec2& uv)
{
    using namespace procedural_texture_detail;

    const float u = fract(uv.x);
    const float v = fract(uv.y);

    if (id == 2)
    {
        const float q = kTwoPi * (8.0f * u
            + 0.35f * sinf(kTwoPi * v)
            + 0.20f * sinf(kTwoPi * (3.0f * u - 2.0f * v)));
        const float w = 0.5f + 0.5f * sinf(q);
        const float oneMinusW = 1.0f - w;
        const float k = oneMinusW * oneMinusW * oneMinusW * oneMinusW;
        const glm::vec3 ivory(0.82f, 0.72f, 0.50f);
        const glm::vec3 blue(0.045f, 0.12f, 0.21f);
        return mix(ivory, blue, k);
    }

    if (id == 3)
    {
        const float scaledV = 8.0f * v;
        const int row = static_cast<int>(floorf(scaledV));
        const float x = 8.0f * u + 0.5f * static_cast<float>(row & 1);
        const int column = static_cast<int>(floorf(x));
        const float fx = fract(x);
        const float fy = fract(scaledV);
        const glm::vec3 teal(0.06f, 0.34f, 0.31f);
        const glm::vec3 terra(0.58f, 0.19f, 0.08f);
        const glm::vec3 alternate = ((column + row) & 1) == 0 ? teal : terra;
        const float diamondDistance = fabsf(fx - 0.5f) + fabsf(fy - 0.5f);
        const float diamond = 1.0f - smoothstep(0.26f, 0.30f, diamondDistance);
        const glm::vec3 gold(0.86f, 0.57f, 0.14f);
        const glm::vec3 diamondBase = mix(alternate, gold, diamond);
        const float edgeDistance = fminf(fminf(fx, 1.0f - fx),
            fminf(fy, 1.0f - fy));
        const float edge = smoothstep(0.025f, 0.05f, edgeDistance);
        const glm::vec3 grout(0.025f, 0.035f, 0.05f);
        return mix(grout, diamondBase, edge);
    }

    return glm::vec3(1.0f);
}

} // namespace renderer

#undef PROCEDURAL_TEXTURE_INLINE
#undef PROCEDURAL_TEXTURE_HD
