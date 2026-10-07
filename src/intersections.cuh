#pragma once

#include "scene_structs.h"

#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>


__host__ __device__ inline glm::vec3 get_point_on_ray(Ray r, float t) {
    return r.org + t * r.dir;
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiply_mv(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param isect_point  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float box_intersection_test(const Geom& box, Ray r, glm::vec3* isect_point,
                                                glm::vec3* normal, bool* outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 1 and is centered at the origin.
 *
 * @param isect_point  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphere_intersection_test(const Geom& sphere, Ray r,
                                                   glm::vec3* isect_point, glm::vec3* normal,
                                                   bool* outside);

__host__ __device__ float plane_intersection_test(const Geom& plane, Ray r, glm::vec3* isect_point,
                                                  glm::vec3* normal, bool* outside);

__device__ bool visible_to_light(Ray r, float light_dist, const Geom* geoms, int num_geoms);

__global__ void compute_intersections(int num_paths, const PathSegment* path_segments,
                                      const Geom* geoms, int num_geoms,
                                      ShadeableIntersection* intersections, MatId* isect_mat_ids);
