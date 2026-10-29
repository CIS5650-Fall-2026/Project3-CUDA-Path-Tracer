#pragma once

// View transform: scene-linear RGB to display-encoded sRGB, applied at
// display and save time only.

#include <cuda_runtime.h>
#include <glm/glm.hpp>

enum ToneMapMode
{
    TONEMAP_NONE = 0,   // clamp only, no gamma
    TONEMAP_ACES = 1,
    TONEMAP_AGX  = 2,
    TONEMAP_AGX_PUNCHY = 3,
};

// 0 = Narkowicz per-channel fit, 1 = Stephen Hill's fit scaled by 1/0.6 on input (three.js convention).
#define ACES_FIT_HILL 0

__host__ __device__ inline float srgbEncode(float c)
{
    c = glm::clamp(c, 0.f, 1.f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}

__host__ __device__ inline glm::vec3 srgbEncode(glm::vec3 c)
{
    return glm::vec3(srgbEncode(c.x), srgbEncode(c.y), srgbEncode(c.z));
}

// Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve" (2016).
__host__ __device__ inline glm::vec3 acesFit(glm::vec3 x)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
}

// Stephen Hill's ACES fit, from ACES.hlsl in MJP's BakingLab (MIT). The
// reference lists the matrices row-major for HLSL, hence the transposes.
__host__ __device__ inline glm::vec3 acesHillFit(glm::vec3 c)
{
    const glm::mat3 inputMat = glm::transpose(glm::mat3(
        0.59719f, 0.35458f, 0.04823f,
        0.07600f, 0.90834f, 0.01566f,
        0.02840f, 0.13383f, 0.83777f));
    const glm::mat3 outputMat = glm::transpose(glm::mat3(
        1.60475f, -0.53108f, -0.07367f,
        -0.10208f, 1.10813f, -0.00605f,
        -0.00327f, -0.07276f, 1.07602f));

    glm::vec3 v = inputMat * (c / 0.6f);
    glm::vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    glm::vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    v = a / b;
    v = outputMat * v;
    return glm::clamp(v, 0.f, 1.f);
}

// AgX after Troy Sobotka, analytic form from Benjamin Wrensch's "Minimal AgX
// Implementation". Matrices are column-major like the reference's GLSL.
__host__ __device__ inline glm::vec3 agxSigmoid(glm::vec3 x)
{
    glm::vec3 x2 = x * x;
    glm::vec3 x4 = x2 * x2;
    return 15.5f * x4 * x2
         - 40.14f * x4 * x
         + 31.96f * x4
         - 6.868f * x2 * x
         + 0.4298f * x2
         + 0.1191f * x
         - 0.00232f;
}

// The "punchy" look from the same reference; Blender ships it under that name.
__host__ __device__ inline glm::vec3 agxLookPunchy(glm::vec3 val)
{
    const glm::vec3 lw(0.2126f, 0.7152f, 0.0722f);
    float luma = glm::dot(val, lw);
    const float power = 1.35f;
    const float sat = 1.4f;
    val = glm::vec3(powf(val.x, power), powf(val.y, power), powf(val.z, power));
    return luma + sat * (val - luma);
}

__host__ __device__ inline glm::vec3 agx(glm::vec3 val, bool punchy)
{
    const glm::mat3 inset(
        0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,
        0.0784335999999992f, 0.878468636469772f, 0.0784336f,
        0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f);
    const glm::mat3 outset(
        1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f,
        -0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f,
        -0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f);
    const float minEv = -12.47393f;
    const float maxEv = 4.026069f;

    val = inset * val;
    val = glm::vec3(log2f(val.x), log2f(val.y), log2f(val.z));
    val = glm::clamp(val, minEv, maxEv);
    val = (val - minEv) / (maxEv - minEv);
    val = agxSigmoid(val);
    if (punchy)
    {
        val = agxLookPunchy(glm::clamp(val, 0.f, 1.f));
    }
    val = outset * val;
    // The sigmoid output is 2.2-gamma encoded; linearize and re-encode as sRGB
    val = glm::clamp(val, 0.f, 1.f);
    val = glm::vec3(powf(val.x, 2.2f), powf(val.y, 2.2f), powf(val.z, 2.2f));
    return srgbEncode(val);
}

__host__ __device__ inline glm::vec3 applyToneMap(glm::vec3 linear, ToneMapMode mode, float exposure)
{
    linear *= exposure;
    switch (mode)
    {
#if ACES_FIT_HILL
    case TONEMAP_ACES: return srgbEncode(acesHillFit(linear));
#else
    case TONEMAP_ACES: return srgbEncode(acesFit(linear));
#endif
    case TONEMAP_AGX:        return agx(linear, false);
    case TONEMAP_AGX_PUNCHY: return agx(linear, true);
    default:           return glm::clamp(linear, 0.f, 1.f);
    }
}
