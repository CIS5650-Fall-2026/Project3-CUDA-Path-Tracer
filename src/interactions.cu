#include "interactions.h"
#include "intersections.h"

#include <cuda_runtime.h>

#include <cmath>

namespace renderer {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

__host__ __device__ bool finiteScalar(float value)
{
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

__host__ __device__ float clampUnit(float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__host__ __device__ bool finite3(const glm::vec3& value)
{
    return finiteScalar(value.x) && finiteScalar(value.y)
        && finiteScalar(value.z);
}

__device__ float srgbToLinear(float raw)
{
    const float value = clampUnit(raw);
    return value <= 0.04045f
        ? value / 12.92f
        : powf((value + 0.055f) / 1.055f, 2.4f);
}

__host__ __device__ void tangentFrame(const glm::vec3& normal,
    glm::vec3* tangent, glm::vec3* bitangent)
{
    const glm::vec3 helper = fabsf(normal.z) < 0.999f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    *tangent = safeNormalize(glm::cross(helper, normal),
        glm::vec3(1.0f, 0.0f, 0.0f));
    *bitangent = glm::cross(normal, *tangent);
}

__device__ int positiveModulo(int value, int modulus)
{
    const int remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

__device__ glm::vec3 fetchTexel(const SceneView& scene,
    const TextureDescriptor& texture, int x, int y, bool decodeSrgb)
{
    x = positiveModulo(x, texture.width);
    y = positiveModulo(y, texture.height);
    const long long local = static_cast<long long>(y) * texture.width + x;
    const long long index = static_cast<long long>(texture.offset) + local;
    if (index < 0 || index >= scene.texelCount) {
        return glm::vec3(0.0f);
    }
    const glm::vec3 raw = scene.texels[index];
    if (!decodeSrgb) {
        return raw;
    }
    return glm::vec3(srgbToLinear(raw.x), srgbToLinear(raw.y),
        srgbToLinear(raw.z));
}

} // namespace

namespace
{

// utilhash is inherited unchanged from the authorized course starter in
// src/intersections.h. The ordered field folding here is original Project 3
// composition: addition and the evolving state make tuple order significant.
__host__ __device__ std::uint32_t foldCounterField(std::uint32_t state,
    std::uint32_t word, std::uint32_t fieldIndex)
{
    const std::uint32_t taggedWord = utilhash(word + fieldIndex
        + fieldIndex * fieldIndex);
    const unsigned int shift = 1U + fieldIndex % 5U;
    return utilhash(state + taggedWord + (state << shift) + fieldIndex);
}

} // namespace

__host__ __device__ std::uint32_t counterRandomWord(std::uint64_t seed,
    int pixel, std::uint64_t sample, int bounce, int dimension)
{
    std::uint32_t state = utilhash(0U);
    state = foldCounterField(state, static_cast<std::uint32_t>(seed), 1U);
    state = foldCounterField(state, static_cast<std::uint32_t>(seed >> 32U), 2U);
    state = foldCounterField(state,
        static_cast<std::uint32_t>(pixel), 3U);
    state = foldCounterField(state, static_cast<std::uint32_t>(sample), 4U);
    state = foldCounterField(state,
        static_cast<std::uint32_t>(sample >> 32U), 5U);
    state = foldCounterField(state,
        static_cast<std::uint32_t>(bounce), 6U);
    state = foldCounterField(state,
        static_cast<std::uint32_t>(dimension), 7U);
    return utilhash(state + 8U);
}

__host__ __device__ float randomFloat(std::uint64_t seed, int pixel,
    std::uint64_t sample, int bounce, int dimension)
{
    // A 23-bit midpoint cannot round to either endpoint in binary32.
    const std::uint32_t mantissa = counterRandomWord(seed, pixel, sample,
        bounce, dimension) >> 9U;
    return (static_cast<float>(mantissa) + 0.5f) * (1.0f / 8388608.0f);
}

__host__ __device__ glm::vec3 safeNormalize(const glm::vec3& value,
    const glm::vec3& fallback)
{
    const float lengthSquared = glm::dot(value, value);
    if (!(lengthSquared > 1.0e-20f) || !finiteScalar(lengthSquared)) {
        return fallback;
    }
    const glm::vec3 result = value * (1.0f / sqrtf(lengthSquared));
    return finite3(result) ? result : fallback;
}

__host__ __device__ glm::vec3 orientedGeometricNormal(const Hit& hit)
{
    const glm::vec3 outward = safeNormalize(hit.geometricNormal,
        glm::vec3(0.0f, 1.0f, 0.0f));
    return hit.frontFace ? outward : -outward;
}

__host__ __device__ glm::vec3 offsetRayOrigin(const glm::vec3& position,
    const glm::vec3& outwardNormal, const glm::vec3& direction)
{
    const float maximum = fmaxf(fabsf(position.x),
        fmaxf(fabsf(position.y), fabsf(position.z)));
    const float distance = 1.0e-4f * fmaxf(1.0f, maximum);
    const float side = glm::dot(direction, outwardNormal) >= 0.0f ? 1.0f : -1.0f;
    return position + outwardNormal * (side * distance);
}

__host__ __device__ glm::vec3 cosineHemisphere(const glm::vec3& normal,
    float u0, float u1)
{
    const float radius = sqrtf(clampUnit(u0));
    const float angle = kTwoPi * clampUnit(u1);
    const float x = radius * cosf(angle);
    const float y = radius * sinf(angle);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));
    glm::vec3 tangent;
    glm::vec3 bitangent;
    tangentFrame(normal, &tangent, &bitangent);
    return safeNormalize(x * tangent + y * bitangent + z * normal, normal);
}

__host__ __device__ float cosineHemispherePdf(const glm::vec3& normal,
    const glm::vec3& direction)
{
    return fmaxf(0.0f, glm::dot(normal, direction)) / kPi;
}

__host__ __device__ float dielectricFresnel(float cosIncident,
    float etaIncident, float etaTransmitted, bool* totalInternalReflection)
{
    if (!(etaIncident > 0.0f) || !(etaTransmitted > 0.0f)
            || !finiteScalar(etaIncident) || !finiteScalar(etaTransmitted)) {
        if (totalInternalReflection != nullptr) {
            *totalInternalReflection = false;
        }
        return 1.0f;
    }
    const float etaScale = fmaxf(etaIncident, etaTransmitted);
    if (fabsf(etaIncident - etaTransmitted) <= 1.0e-7f * etaScale) {
        if (totalInternalReflection != nullptr) {
            *totalInternalReflection = false;
        }
        return 0.0f;
    }
    cosIncident = clampUnit(fabsf(cosIncident));
    const float eta = etaIncident / etaTransmitted;
    const float sinTransmittedSquared = eta * eta
        * fmaxf(0.0f, 1.0f - cosIncident * cosIncident);
    if (sinTransmittedSquared >= 1.0f) {
        if (totalInternalReflection != nullptr) {
            *totalInternalReflection = true;
        }
        return 1.0f;
    }
    if (totalInternalReflection != nullptr) {
        *totalInternalReflection = false;
    }
    const float cosTransmitted = sqrtf(fmaxf(0.0f,
        1.0f - sinTransmittedSquared));
    const float parallelNumerator = etaTransmitted * cosIncident
        - etaIncident * cosTransmitted;
    const float parallelDenominator = etaTransmitted * cosIncident
        + etaIncident * cosTransmitted;
    const float perpendicularNumerator = etaIncident * cosIncident
        - etaTransmitted * cosTransmitted;
    const float perpendicularDenominator = etaIncident * cosIncident
        + etaTransmitted * cosTransmitted;
    const float parallel = fabsf(parallelDenominator) > 1.0e-20f
        ? parallelNumerator / parallelDenominator : 1.0f;
    const float perpendicular = fabsf(perpendicularDenominator) > 1.0e-20f
        ? perpendicularNumerator / perpendicularDenominator : 1.0f;
    return clampUnit(0.5f * (parallel * parallel
        + perpendicular * perpendicular));
}

__host__ __device__ bool refractDirection(const glm::vec3& incident,
    const glm::vec3& normalAgainstIncident, float etaIncident,
    float etaTransmitted, glm::vec3* transmitted)
{
    const float eta = etaIncident / etaTransmitted;
    const float cosIncident = fminf(1.0f,
        fmaxf(0.0f, glm::dot(-incident, normalAgainstIncident)));
    const float discriminant = 1.0f - eta * eta
        * fmaxf(0.0f, 1.0f - cosIncident * cosIncident);
    if (discriminant <= 0.0f) {
        return false;
    }
    *transmitted = safeNormalize(eta * incident
        + (eta * cosIncident - sqrtf(discriminant)) * normalAgainstIncident,
        -normalAgainstIncident);
    return true;
}

__host__ __device__ float powerHeuristic(float firstPdf, float secondPdf)
{
    firstPdf = fmaxf(0.0f, firstPdf);
    secondPdf = fmaxf(0.0f, secondPdf);
    if (!finiteScalar(firstPdf) || !finiteScalar(secondPdf)) {
        const bool firstPositiveInfinity = firstPdf == firstPdf
            && firstPdf > FLT_MAX;
        const bool secondPositiveInfinity = secondPdf == secondPdf
            && secondPdf > FLT_MAX;
        if (firstPositiveInfinity && secondPositiveInfinity) {
            return 0.5f;
        }
        return firstPositiveInfinity ? 1.0f : 0.0f;
    }
    const float scale = fmaxf(firstPdf, secondPdf);
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    const float firstScaled = firstPdf / scale;
    const float secondScaled = secondPdf / scale;
    const float firstSquared = firstScaled * firstScaled;
    const float secondSquared = secondScaled * secondScaled;
    const float denominator = firstSquared + secondSquared;
    return denominator > 0.0f ? firstSquared / denominator : 0.0f;
}

__device__ glm::vec3 sampleTexture(const SceneView& scene, int textureId,
    const glm::vec2& uv, bool decodeSrgb)
{
    if (textureId < 0 || textureId >= scene.textureCount
            || scene.textures == nullptr || scene.texels == nullptr) {
        return glm::vec3(1.0f);
    }
    const TextureDescriptor texture = scene.textures[textureId];
    if (texture.width <= 0 || texture.height <= 0 || texture.offset < 0) {
        return glm::vec3(1.0f);
    }

    const float repeatedU = uv.x - floorf(uv.x);
    const float repeatedV = uv.y - floorf(uv.y);
    const float x = repeatedU * texture.width - 0.5f;
    const float y = repeatedV * texture.height - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - x0;
    const float ty = y - y0;
    const glm::vec3 c00 = fetchTexel(scene, texture, x0, y0, decodeSrgb);
    const glm::vec3 c10 = fetchTexel(scene, texture, x0 + 1, y0, decodeSrgb);
    const glm::vec3 c01 = fetchTexel(scene, texture, x0, y0 + 1, decodeSrgb);
    const glm::vec3 c11 = fetchTexel(scene, texture, x0 + 1, y0 + 1, decodeSrgb);
    return glm::mix(glm::mix(c00, c10, tx), glm::mix(c01, c11, tx), ty);
}

__device__ float sampleHeight(const SceneView& scene, int textureId,
    const glm::vec2& uv)
{
    const glm::vec3 value = sampleTexture(scene, textureId, uv, false);
    return glm::dot(value, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

__device__ SurfaceEvaluation evaluateSurface(const SceneView& scene,
    const Material& material, const Hit& hit)
{
    SurfaceEvaluation result;
    result.baseColor = glm::max(material.color, glm::vec3(0.0f));
    if (material.textureId >= 0) {
        result.baseColor *= sampleTexture(scene, material.textureId, hit.uv, true);
    }
    if (material.procedural != 0) {
        const float scale = material.checkerScale > 0.0f
            ? material.checkerScale : 8.0f;
        const int checker = static_cast<int>(floorf(hit.uv.x * scale))
            + static_cast<int>(floorf(hit.uv.y * scale));
        result.baseColor *= (checker & 1) == 0 ? 1.0f : 0.18f;
    }
    result.emission = result.baseColor * fmaxf(0.0f, material.emittance);

    const glm::vec3 geometric = orientedGeometricNormal(hit);
    glm::vec3 shading = safeNormalize(hit.shadingNormal, geometric);
    if (glm::dot(shading, geometric) < 0.0f) {
        shading = -shading;
    }

    if (material.bumpTextureId >= 0 && material.bumpStrength != 0.0f) {
        const TextureDescriptor texture =
            material.bumpTextureId < scene.textureCount && scene.textures != nullptr
            ? scene.textures[material.bumpTextureId] : TextureDescriptor{};
        const float du = texture.width > 0 ? 1.0f / texture.width : 1.0e-3f;
        const float dv = texture.height > 0 ? 1.0f / texture.height : 1.0e-3f;
        const float derivativeU = (sampleHeight(scene, material.bumpTextureId,
            hit.uv + glm::vec2(du, 0.0f)) - sampleHeight(scene,
            material.bumpTextureId, hit.uv - glm::vec2(du, 0.0f)))
            / (2.0f * du);
        const float derivativeV = (sampleHeight(scene, material.bumpTextureId,
            hit.uv + glm::vec2(0.0f, dv)) - sampleHeight(scene,
            material.bumpTextureId, hit.uv - glm::vec2(0.0f, dv)))
            / (2.0f * dv);

        // A constant height field must leave an interpolated smooth normal
        // exactly unchanged. For a varying field, retain the world-space
        // magnitudes of dp/du and dp/dv while first projecting them into the
        // current shading tangent plane.
        if (finiteScalar(derivativeU) && finiteScalar(derivativeV)
                && (derivativeU != 0.0f || derivativeV != 0.0f)) {
            const glm::vec3 tangentU = hit.dpdu
                - shading * glm::dot(shading, hit.dpdu);
            const glm::vec3 tangentV = hit.dpdv
                - shading * glm::dot(shading, hit.dpdv);
            const glm::vec3 baseCross = glm::cross(tangentU, tangentV);
            const float baseLengthSquared = glm::dot(baseCross, baseCross);
            if (baseLengthSquared > 1.0e-20f && finite3(tangentU)
                    && finite3(tangentV) && finite3(baseCross)) {
                const glm::vec3 perturbedU = tangentU + shading
                    * (material.bumpStrength * derivativeU);
                const glm::vec3 perturbedV = tangentV + shading
                    * (material.bumpStrength * derivativeV);
                glm::vec3 bumped = safeNormalize(
                    glm::cross(perturbedU, perturbedV), shading);
                if (glm::dot(bumped, shading) < 0.0f) {
                    bumped = -bumped;
                }
                if (finite3(bumped)
                        && glm::dot(bumped, geometric) > 1.0e-5f) {
                    shading = bumped;
                }
            }
        }
    }

    result.shadingNormal = shading;
    return result;
}

} // namespace renderer
