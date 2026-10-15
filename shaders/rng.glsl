#ifndef RNG_GLSL
#define RNG_GLSL

#include "math.glsl"

void initSeed(uvec3 launchId, uvec3 launchSize, uint renderedFrameCount);
uint utilhash(uint a);
float rng();

uvec2 gSeed;
void initSeed(uvec3 launchId, uvec3 launchSize, uint renderedFrameCount)
{
    uint pixelIndex = launchId.y * launchSize.x + launchId.x;
    gSeed = uvec2(utilhash(pixelIndex), utilhash(renderedFrameCount));
}

// Handy-dandy hash function that provides seeds for random number generation.
uint utilhash(uint a)
{
    a = (a + 0x7ed55d16U) + (a << 12);
    a = (a ^ 0xc761c23cU) ^ (a >> 19);
    a = (a + 0x165667b1U) + (a << 5);
    a = (a + 0xd3a2646cU) ^ (a << 9);
    a = (a + 0xfd7046c5U) + (a << 3);
    a = (a ^ 0xb55a4f09U) ^ (a >> 16);
    return a;
}

// Adapted from ShaderToy https://www.shadertoy.com/view/4tXyWN
float rng()
{
    gSeed += uvec2(1);
    uvec2 q = 1103515245U * ((gSeed >> 1U) ^ (gSeed.yx));
    uint  n = 1103515245U * ((q.x) ^ (q.y >> 3U));

    // Return values in [0, 1)
    return min(float(n) * (1.0 / float(0xffffffffU)), kOneMinusEpsilon);
}

#endif