#pragma once

#include "geometry.h"

#include <cstdint>

namespace renderer {

struct SurfaceEvaluation
{
    glm::vec3 baseColor{1.0f};
    glm::vec3 emission{0.0f};
    glm::vec3 shadingNormal{0.0f, 1.0f, 0.0f};
};

// Counter-based random samples. Every value is a pure function of the run
// seed, original pixel, global sample number, bounce, and named dimension.
__host__ __device__ std::uint32_t counterRandomWord(std::uint64_t seed,
    int pixel, std::uint64_t sample, int bounce, int dimension);
__host__ __device__ float randomFloat(std::uint64_t seed, int pixel,
    std::uint64_t sample, int bounce, int dimension);

__host__ __device__ glm::vec3 safeNormalize(const glm::vec3& value,
    const glm::vec3& fallback);
__host__ __device__ glm::vec3 orientedGeometricNormal(const Hit& hit);
__host__ __device__ glm::vec3 offsetRayOrigin(const glm::vec3& position,
    const glm::vec3& outwardNormal, const glm::vec3& direction);
__host__ __device__ glm::vec3 cosineHemisphere(const glm::vec3& normal,
    float u0, float u1);
__host__ __device__ float cosineHemispherePdf(const glm::vec3& normal,
    const glm::vec3& direction);
__host__ __device__ float dielectricFresnel(float cosIncident,
    float etaIncident, float etaTransmitted, bool* totalInternalReflection);
__host__ __device__ bool refractDirection(const glm::vec3& incident,
    const glm::vec3& normalAgainstIncident, float etaIncident,
    float etaTransmitted, glm::vec3* transmitted);
__host__ __device__ float powerHeuristic(float firstPdf, float secondPdf);

// Scene stores normalized source texels. Color lookup decodes each fetched
// texel from sRGB before interpolation; height lookup retains the raw samples.
__device__ glm::vec3 sampleTexture(const SceneView& scene, int textureId,
    const glm::vec2& uv, bool decodeSrgb);
__device__ float sampleHeight(const SceneView& scene, int textureId,
    const glm::vec2& uv);
__device__ SurfaceEvaluation evaluateSurface(const SceneView& scene,
    const Material& material, const Hit& hit);

} // namespace renderer
