#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// ---------------------------------------------------------------------------
// Helpers (file-local)
// ---------------------------------------------------------------------------
#define USE_SCHLICK 0      // 1 = Schlick approximation, 0 = exact dielectric Fresnel
#define RAY_EPS 1e-3f

__host__ __device__ static void makeBasis(const glm::vec3& n, glm::vec3& t, glm::vec3& b)
{
    glm::vec3 up = (fabsf(n.x) < 0.9f) ? glm::vec3(1.f, 0.f, 0.f) : glm::vec3(0.f, 1.f, 0.f);
    t = glm::normalize(glm::cross(up, n));
    b = glm::cross(n, t);
}

// Phong lobe around the perfect reflection direction r
// (GPU Gems 3 ch.20, eqs 7-9: cos(theta) = xi1^(1/(n+1)), phi = 2*pi*xi2).
__host__ __device__ static glm::vec3 samplePhongLobe(
    const glm::vec3& r, float exponent, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    float u1 = u01(rng), u2 = u01(rng);
    float cosT = powf(u1, 1.f / (exponent + 1.f));
    float sinT = sqrtf(fmaxf(0.f, 1.f - cosT * cosT));
    float phi = TWO_PI * u2;
    glm::vec3 t, b;
    makeBasis(r, t, b);
    return glm::normalize(t * (sinT * cosf(phi)) + b * (sinT * sinf(phi)) + r * cosT);
}

// Dielectric reflectance. cosI = dot(-incident, normal facing incident side).
// n1 = IOR on incident side, n2 = IOR on far side. Returns 1 on total internal reflection.
__host__ __device__ static float fresnelDielectric(float cosI, float n1, float n2)
{
    cosI = glm::clamp(cosI, 0.f, 1.f);
    float eta = n2 / n1;
    float sin2T = (1.f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.f) return 1.f;
    float cosT = sqrtf(1.f - sin2T);
#if USE_SCHLICK
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float c = (n1 > n2) ? cosT : cosI;
    float m = 1.f - c;
    return r0 + (1.f - r0) * m * m * m * m * m;
#else
    float rParl = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rParl * rParl + rPerp * rPerp);
#endif
}

// ---------------------------------------------------------------------------
// scatterRay
// ---------------------------------------------------------------------------
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    const glm::vec3 I = glm::normalize(pathSegment.ray.direction);
    glm::vec3 newDir;
    glm::vec3 throughput(1.f);

    if (m.hasRefractive > 0.f)
    {
        // Glass / water: Fresnel chooses reflect vs. refract.
        float ior = (m.indexOfRefraction > 0.f) ? m.indexOfRefraction : 1.5f;
        float n1 = outside ? 1.f : ior;
        float n2 = outside ? ior : 1.f;

        float F = fresnelDielectric(glm::dot(-I, normal), n1, n2);
        glm::vec3 refr = glm::refract(I, normal, n1 / n2);
        bool tir = glm::dot(refr, refr) < 1e-8f;

        if (tir || u01(rng) < F)
        {
            newDir = glm::reflect(I, normal);
        }
        else
        {
            newDir = glm::normalize(refr);
            if (outside) throughput = m.color;   // tint once, when entering
        }
    }
    else if (m.hasReflective > 0.f && u01(rng) < m.hasReflective)
    {
        // Specular: mirror, or Phong lobe if exponent > 0.
        glm::vec3 r = glm::reflect(I, normal);
        newDir = (m.specular.exponent > 0.f) ? samplePhongLobe(r, m.specular.exponent, rng) : r;

        if (glm::dot(newDir, normal) <= 0.f)
        {
            pathSegment.color = glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
            return;
        }
        throughput = (glm::dot(m.specular.color, m.specular.color) > 0.f) ? m.specular.color : m.color;
    }
    else
    {
        // Diffuse
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        throughput = m.color;
    }

    pathSegment.color *= throughput;

    // Offset to the side the new ray travels into (crosses the surface for refraction).
    float side = (glm::dot(newDir, normal) > 0.f) ? 1.f : -1.f;
    pathSegment.ray.origin = intersect + side * normal * RAY_EPS;
    pathSegment.ray.direction = newDir;
    pathSegment.remainingBounces--;
}
