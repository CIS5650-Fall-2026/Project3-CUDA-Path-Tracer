#pragma once

#include <cuda_runtime.h>
#include "utilities.h"

/*
* Intended for making basic procedural textures
*/

__host__ __device__ unsigned int hash(int x, int y, int z);

__host__ __device__ float hashFloat(unsigned int h);

__host__ __device__ glm::vec3 hashVec(int x, int y, int z);

/**
* Perlin noise
*/
__host__ __device__ float perlin(glm::vec3 p);

/**
* FBM noise
*/
__host__ __device__ float fbm(glm::vec3 p, int octaves);

__host__ __device__ glm::vec3 fbmVec(glm::vec3 p, int octaves);

/**
* Voronoi noise
*/
__host__ __device__ float voronoi(glm::vec3 p);

__host__ __device__ float voronoiBetter(glm::vec3 p, float jitter, unsigned int& cellId);