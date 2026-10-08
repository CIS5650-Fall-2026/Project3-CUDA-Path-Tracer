#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>


__host__ __device__ void sampleRandomLight(Geom* geo, int num_geoms, VolumeData& vd, const Light* lights, int num_lights, const Material* materials, glm::vec3 intersect, glm::vec3 normal, glm::vec3 wo, const Material& m, thrust::default_random_engine& rng, glm::vec3& Li);
