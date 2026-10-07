#pragma once

#include "light_sampler.cuh"
#include "scene_structs.h"

__global__ void shade_material(int iter, int num_paths, int depth,
                               const ShadeableIntersection* shadeable_intersections,
                               const MatId* isect_mat_ids, PathSegment* path_segments,
                               ShadowRay* shadow_rays, const Material* materials,
                               LightSampler light_sampler, const Geom* geoms, glm::vec3* image);

__global__ void trace_shadow_rays(int num_srays, int num_geoms, ShadowRay* shadow_rays,
                                  const Geom* geoms, glm::vec3* image);
