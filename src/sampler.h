#pragma once

#include "featureToggles.h"
#include "sceneStructs.h"

#include <cmath>

// Radical inverse in the given prime base. Sample index n maps to a point in [0, 1).
__host__ __device__ inline float radicalInverse(unsigned int n, unsigned int base)
{
    float invBase = 1.0f / static_cast<float>(base);
    float invPow = invBase;
    float result = 0.0f;
    while (n > 0u) {
        unsigned int digit = n % base;
        result += static_cast<float>(digit) * invPow;
        n /= base;
        invPow *= invBase;
    }
    return result;
}

// Next Halton coordinate for this path, shifted per pixel (Cranley-Patterson).
// Dimension 0 is the first sample consumed on the camera ray.
__host__ __device__ inline float sample1D(PathSegment& path)
{
    // One prime per dimension. Past this, high-dimensional Halton correlates,
    // so further samples come from a hashed value instead of a repeated base.
    const unsigned int primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131
    };
    const int numPrimes = 32;

    int dim = path.dimension;
    path.dimension = dim + 1;

    float u;
    if (dim >= 0 && dim < numPrimes) {
        u = radicalInverse(static_cast<unsigned int>(path.sampleIndex), primes[dim]);
    } else {
        // in x = a*x + c, 1664525 is a, the multiplier, and 1013904223 is c
        unsigned int x = static_cast<unsigned int>(path.sampleIndex) * 1664525u
            + static_cast<unsigned int>(dim) * 1013904223u;
        x ^= x >> 16;
        // turns a 32-bit int into a sample in [0, 1) 
        // 0x00ffffffu keeps the lowest 24 bits , which means the value is 0 - 16777216
        // spreading ints across [0, 1) in steps of 1/2^24
        u = (x & 0x00ffffffu) * (1.0f / 16777216.0f);
    }

    float shifted = u + path.scramble;
    return shifted - floorf(shifted);
}
