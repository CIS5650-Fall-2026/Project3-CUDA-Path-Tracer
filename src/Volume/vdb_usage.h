#pragma once

#include <thrust/random.h>

#include "../sceneStructs.h"

#include <glm/glm.hpp>

__host__ __device__ float sampleTemperature(glm::vec3 pos, VolumeData& vd, const Volume& volume);

__host__ __device__  float vdbIntersectionTest(VolumeData& vd, const Volume& volume, Ray ray, thrust::default_random_engine& rng, float tMax);
