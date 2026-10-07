#pragma once

#include "bvh.h"
#include "sceneStructs.h"

#include <thrust/random.h>

namespace Nee
{
__device__ float power_heuristic(float pdfA, float pdfB);

__device__ bool shadow_ray_visible(
    const glm::vec3& origin,
    const NeeSample& neeSample,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    int triangles_size,
    BVHNode* bvh,
    Material* materials,
    int materials_size);

__device__ glm::vec3 evaluate_emissive_hit(
    const PathSegment& pathSegment,
    glm::vec3 hitPoint,
    glm::vec3 lightNormal,
    int triangleId,
    const Material& lightMaterial,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    float totalLightArea);

__device__ LightSample sample_light(
    SceneLight* lights,
    int lights_size,
    float totalLightArea,
    thrust::default_random_engine& rng);

__device__ LightPointSample sample_point_from_light(
    const LightSample& lightSample,
    Triangle* triangles,
    int triangles_size,
    thrust::default_random_engine& rng);

__device__ NeeSample get_nee(
    const glm::vec3& surfacePoint,
    const glm::vec3& surfaceNormal,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    Material* materials,
    int materials_size,
    float totalLightArea,
    thrust::default_random_engine& rng);

__device__ float pdf_light_for_triangle_hit(
    const glm::vec3& surfacePoint,
    const glm::vec3& lightPoint,
    const glm::vec3& lightNormal,
    int triangleId,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    const Material& lightMaterial,
    float totalLightArea);
}
