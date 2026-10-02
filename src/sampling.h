#pragma once
#include "glm/glm.hpp"
#include "intersections.h"
#include <cmath>
#include <cuda_runtime.h>

// Stateless samples: reordering/compacting paths or resuming does not change them.
__host__ __device__ inline unsigned int sampleHash(unsigned int x)
{
    return utilhash(x); // Hash the sample key without maintaining RNG state.
}
__host__ __device__ inline float unitFloat(unsigned int x)
{
    return (x >> 8) * (1.0f / 16777216.0f);
}
struct Sampler
{
    unsigned int pixel, sample, seed;
    bool halton;
    const int *primes;
    __host__ __device__ float at(int dimension) const
    {
        unsigned int key = sampleHash(pixel ^ sampleHash(seed) ^ sampleHash(dimension + 1));
        if (!halton)
            return unitFloat(sampleHash(key ^ sampleHash(sample + 1)));
        unsigned int index = sample;
        unsigned int base = primes[dimension];
        float inv = 1.0f / base, factor = inv, value = 0;
        unsigned int digitPosition = 0;
        // A prime-base affine digit permutation is bijective. Scramble zero
        // digits too, so samples of different lengths share the same sequence.
        while (factor > 1.0e-7f)
        {
            unsigned int h = sampleHash(key ^ sampleHash(0x9e3779b9u + digitPosition++));
            unsigned int multiplier = 1 + h % (base - 1);
            unsigned int shift = (h >> 16) % base;
            unsigned int digit = (multiplier * (index % base) + shift) % base;
            value += digit * factor;
            index /= base;
            factor *= inv;
        }
        // A separate Cranley-Patterson rotation for every pixel and dimension.
        value += unitFloat(key);
        return fminf(value - floorf(value), 0.99999994f);
    }
};
__host__ __device__ inline float powerWeight(float a, float b)
{
    if (a <= 0)
        return 0;
    float ratio = b / a;
    return 1.0f / (1.0f + ratio * ratio);
}
