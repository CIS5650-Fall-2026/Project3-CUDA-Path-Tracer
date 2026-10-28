#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// Blend the four texels around the sample point instead of taking the nearest one
#define TEXTURE_BILINEAR 1

/**
 * Looks up the color of texture `tex` at `uv`, nearest texel or bilinear
 * depending on TEXTURE_BILINEAR
 */
__host__ __device__ glm::vec3 sampleTexture(
    const glm::vec3* texels,
    const TextureInfo& tex,
    glm::vec2 uv);

/**
 * Computes a tile-and-grout pattern at `uv` from the material's tile parameters
 */
__host__ __device__ glm::vec3 proceduralTiles(
    const Material& m,
    glm::vec2 uv);

/**
 * Height of the tile pattern at `uv`: 1 on a tile, 0 at the bottom of the grout,
 * with a short ramp between them
 */
__host__ __device__ float proceduralTilesHeight(
    const Material& m,
    glm::vec2 uv);

/**
 * Tilts the shading normal by the slope of the material's height pattern
 *
 * @param normal   geometric normal at the hit, world space
 * @param tangent  direction of increasing u at the hit, world space
 */
__host__ __device__ glm::vec3 bumpNormal(
    glm::vec3 normal,
    glm::vec3 tangent,
    const Material& m,
    glm::vec2 uv);

/**
 * Picks a point uniformly by area on the surface of an emissive box
 *
 * @param point   Output parameter, world-space position on the light
 * @param normal  Output parameter, outward normal of the light at that point
 * @param area    Output parameter, total surface area of the box, world units
 */
__host__ __device__ void sampleBoxLight(
    const Geom& light,
    thrust::default_random_engine& rng,
    glm::vec3& point,
    glm::vec3& normal,
    float& area);

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 geomNormal,
    const Material& m,
    thrust::default_random_engine& rng);
