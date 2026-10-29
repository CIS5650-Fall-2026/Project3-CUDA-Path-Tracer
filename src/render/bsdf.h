#pragma once

// glTF metallic-roughness BSDF (glTF 2.0 spec Appendix B plus the clearcoat,
// specular, ior, transmission and volume extensions), sampled one lobe at a time.
// Transmission is a solid boundary with one roughness for both sides;
// volume only adds absorption, applied by the shade kernel.

#include "render/pbr_surface.h"
#include "scene/sceneStructs.h"

#include <glm/glm.hpp>
#include <thrust/random.h>

// Pushes hitPoint off along normal by EPSILON scaled with its magnitude: a
// fixed EPSILON is below one float ulp past coordinates of about 170.
__host__ __device__ glm::vec3 offsetOrigin(glm::vec3 hitPoint, glm::vec3 normal);

// Samples one lobe: sets path.ray, multiplies path.color by f * cos / pdf,
// sets path.pdf (0 for a smooth lobe). False when the sample carries no
// light; path is then unchanged. surfaceNormal is the unmapped normal facing
// the incoming ray.
__host__ __device__ bool scatterPbr(
    PathSegment& path,
    glm::vec3 hitPoint,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s,
    thrust::default_random_engine& rng);

struct BsdfEval
{
    glm::vec3 fCos;  // f(wo, wi) * |cos|, summed over the lobes
    float pdf;       // solid-angle density scatterPbr picks wi with
};

// f * cos and pdf at (wo, wi) over the rough lobes; smooth lobes add nothing.
// wo points back along the incoming ray, wi away from the hit, both unit length.
__host__ __device__ BsdfEval evalPbr(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s);
