#pragma once

// Environment lookup and sampling on the device (tex2D, kernels only).

#include <cuda_runtime.h>

#include "render/textures.h"
#include "utilities.h"

#include <glm/glm.hpp>

// World direction into the map's frame: a rotation about +Y.
__device__ __forceinline__ glm::vec3 environmentToMap(const EnvironmentMap& env, glm::vec3 direction)
{
    const float c = env.cosRotation;
    const float s = env.sinRotation;
    return glm::vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z);
}

__device__ __forceinline__ glm::vec3 environmentToWorld(const EnvironmentMap& env, glm::vec3 d)
{
    const float c = env.cosRotation;
    const float s = env.sinRotation;
    return glm::vec3(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

// Lat-long coordinates of a map direction: u around +Y from -X, v down from +Y.
__device__ __forceinline__ glm::vec2 environmentUv(glm::vec3 d)
{
    const float u = 0.5f + atan2f(d.z, d.x) * (1.0f / TWO_PI);
    const float v = acosf(glm::clamp(d.y, -1.0f, 1.0f)) * (1.0f / PI);
    return glm::vec2(u, v);
}

__device__ __forceinline__ glm::vec3 environmentRadiance(const EnvironmentMap& env, glm::vec3 direction)
{
    if (env.texture == 0)
    {
        return env.radiance;
    }
    const glm::vec2 uv = environmentUv(environmentToMap(env, direction));
    const float4 texel = tex2D<float4>(env.texture, uv.x, uv.y);
    return env.radiance * glm::vec3(texel.x, texel.y, texel.z);
}

// First index whose running sum exceeds u; an entry that adds nothing is never returned.
__device__ __forceinline__ int searchCdf(const float* cdf, int n, float u)
{
    int lo = 0;
    int hi = n - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;
        if (cdf[mid] > u)
        {
            hi = mid;
        }
        else
        {
            lo = mid + 1;
        }
    }
    return lo;
}

// Unit-square density to solid angle: d omega = 2 pi^2 sin(theta) du dv, sin kept off zero at the poles.
__device__ __forceinline__ float environmentPdfFromUv(float pdfUv, float cosTheta)
{
    const float sinTheta = fmaxf(sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta)), 1e-8f);
    return pdfUv / (2.0f * PI * PI * sinTheta);
}

// Solid-angle density of environmentSample for a world direction, before the light pick probability.
__device__ __forceinline__ float environmentPdf(const EnvironmentMap& env, glm::vec3 direction)
{
    if (env.pdfUv == nullptr)
    {
        return 1.0f / (4.0f * PI);
    }
    const glm::vec3 d = environmentToMap(env, direction);
    const glm::vec2 uv = environmentUv(d);
    const int i = min((int)(uv.x * env.width), env.width - 1);
    const int j = min((int)(uv.y * env.height), env.height - 1);
    return environmentPdfFromUv(env.pdfUv[j * env.width + i], d.y);
}

// Draws a world direction from the marginal and conditional CDFs, jittered
// within the cell. False only when float rounding lands on an empty cell.
__device__ __forceinline__ bool environmentSample(const EnvironmentMap& env, float u1, float u2, glm::vec3& direction,
    float& pdf)
{
    if (env.pdfUv == nullptr)
    {
        // Uniform over the sphere
        const float y = 1.0f - 2.0f * u1;
        const float r = sqrtf(fmaxf(0.0f, 1.0f - y * y));
        const float phi = TWO_PI * u2;
        direction = glm::vec3(r * cosf(phi), y, r * sinf(phi));
        pdf = 1.0f / (4.0f * PI);
        return true;
    }
    const int j = searchCdf(env.marginalCdf, env.height, u1);
    const float* rowCdf = env.conditionalCdf + j * env.width;
    const int i = searchCdf(rowCdf, env.width, u2);
    const float pdfUv = env.pdfUv[j * env.width + i];
    if (!(pdfUv > 0.0f))
    {
        return false;
    }
    const float rowBelow = j > 0 ? env.marginalCdf[j - 1] : 0.0f;
    const float columnBelow = i > 0 ? rowCdf[i - 1] : 0.0f;
    const float dv = (u1 - rowBelow) / (env.marginalCdf[j] - rowBelow);
    const float du = (u2 - columnBelow) / (rowCdf[i] - columnBelow);
    const float v = (j + dv) / env.height;
    const float u = (i + du) / env.width;
    const float theta = v * PI;
    const float phi = (u - 0.5f) * TWO_PI;
    const float sinTheta = sinf(theta);
    const glm::vec3 d(sinTheta * cosf(phi), cosf(theta), sinTheta * sinf(phi));
    direction = environmentToWorld(env, d);
    pdf = environmentPdfFromUv(pdfUv, d.y);
    return true;
}
