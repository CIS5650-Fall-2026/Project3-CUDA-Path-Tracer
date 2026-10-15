#ifndef SAMPLE_WARPING_GLSL
#define SAMPLE_WARPING_GLSL

#include "math.glsl"

vec3 squareToDiskConcentric(vec2 xi)
{
    vec2 uOffset = 2.f * xi - vec2(1.f, 1.f);

    if (areEqual(uOffset.x, 0.f)
        && areEqual(uOffset.y, 0.f))
    {
        return vec3(0.f, 0.f, 0.f);
    }

    float theta, r;

    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = kPi * 0.25f * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = (kPi * 0.5f) - (kPi * 0.25f) * (uOffset.x / uOffset.y);
    }

    return r * vec3(cos(theta), sin(theta), 0.f);
}

vec3 squareToHemisphereCosine(vec2 xi)
{
    vec3 uniformDisk = squareToDiskConcentric(xi);
    float z = sqrt(max(0.f, 1.f - (uniformDisk.x * uniformDisk.x) - (uniformDisk.y * uniformDisk.y)));
    return vec3(uniformDisk.x, uniformDisk.y, z);
}

float squareToHemisphereCosinePDF(vec3 sampleVec)
{
    return abs(sampleVec.z) * kInvPi;
}

#endif
