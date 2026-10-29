#pragma once

// Sequential trace through a lens prescription, host and device.
// Lens space is millimeters: z runs from the front vertex (z = 0) toward the
// film (z > 0), the scene is at z < 0. A positive radius bulges toward the scene.

#include "scene/lens.h"

#include <glm/glm.hpp>

#include <cmath>

// The lens as the generate kernel sees it; surfaces is null without a lens.
struct LensDevice
{
    const LensSurface* surfaces;
    int count;
    float filmDistance;     // front vertex to film, mm
    float entrancePupilZ;   // mm behind the front vertex; Camera::position sits here
    float rearZ;            // rear vertex, mm behind the front vertex
    float aimRadius;        // camera rays are aimed within this disk on the rear vertex's plane (lensAimRadius)
    float pixelMm;          // film millimeters per pixel
    float mmToScene;        // 0.001 * UNITS_PER_METER
    // Exit pupil table (exit_pupil.h): pupilBinCount boxes over film radii 0 to halfDiagonal.
    const glm::vec4* pupil;
    int pupilBinCount;
    float halfDiagonal;
    int useExitPupil;       // 0: aim at the whole rear element
    float weightNorm;       // divides the camera weight: the center pixel's pupil area, or film-to-rear distance squared
    int* blocked;           // count of blocked camera rays; null = do not count
};

// Hit of the ray on surface s (vertex at z) with the unit normal facing the ray.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool intersectLensSurface(const LensSurface& s, float z, glm::vec3 o, glm::vec3 d, glm::vec3& p, glm::vec3& n)
{
    if (s.radius == 0.0f)
    {
        if (d.z == 0.0f)
        {
            return false;
        }
        const float t = (z - o.z) / d.z;
        if (t <= 0.0f)
        {
            return false;
        }
        p = o + t * d;
        n = glm::vec3(0.0f, 0.0f, d.z > 0.0f ? -1.0f : 1.0f);
        return true;
    }
    const float R = s.radius;
    const glm::vec3 c(0.0f, 0.0f, z + R);
    const glm::vec3 oc = o - c;
    const float b = glm::dot(oc, d);
    const float cc = glm::dot(oc, oc) - R * R;
    const float disc = b * b - cc;
    if (disc < 0.0f)
    {
        return false;
    }
    const float sq = sqrtf(disc);
    // The surface is the half of the sphere around the vertex, which picks the root
    const bool nearer = (d.z > 0.0f) == (R > 0.0f);
    const float t = nearer ? -b - sq : -b + sq;
    if (t <= 0.0f)
    {
        return false;
    }
    p = o + t * d;
    n = (p - c) / fabsf(R);
    if (glm::dot(n, d) > 0.0f)
    {
        n = -n;
    }
    return true;
}

// Snell's law from index n1 into n2; false at total internal reflection.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool refractLens(glm::vec3 d, glm::vec3 n, float n1, float n2, glm::vec3& out)
{
    const float eta = n1 / n2;
    const float cosI = -glm::dot(n, d);
    const float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T >= 1.0f)
    {
        return false;
    }
    out = eta * d + (eta * cosI - sqrtf(1.0f - sin2T)) * n;
    return true;
}

// Traces o, d through surfaces from to to (either direction); false when the
// lens blocks the ray. Aspheric terms are not applied yet: every surface is
// traced as the sphere of its paraxial radius.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool traceLens(const LensSurface* s, int from, int to, float zFrom, glm::vec3& o, glm::vec3& d)
{
    const int step = to >= from ? 1 : -1;
    float z = zFrom;
    for (int i = from;; i += step)
    {
        const LensSurface& surf = s[i];
        glm::vec3 p;
        glm::vec3 n;
        if (!intersectLensSurface(surf, z, o, d, p, n))
        {
            return false;
        }
        if (p.x * p.x + p.y * p.y > surf.apertureRadius * surf.apertureRadius)
        {
            return false;
        }
        o = p;
        if (!surf.isStop)
        {
            // ior is the glass behind the surface; air before the first
            const float sceneSide = i > 0 ? s[i - 1].ior : 1.0f;
            const float n1 = step > 0 ? sceneSide : surf.ior;
            const float n2 = step > 0 ? surf.ior : sceneSide;
            if (n1 != n2 && !refractLens(d, n, n1, n2, d))
            {
                return false;
            }
        }
        if (i == to)
        {
            return true;
        }
        z += step > 0 ? surf.thickness : -s[i - 1].thickness;
    }
}
