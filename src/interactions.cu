#include "interactions.h"

#include "utilities.h"

#include <glm/gtx/norm.hpp>
#include <thrust/random.h>

// has to beat the 1e-4 backoff in getPointOnRay, or transmitted rays
// restart on the wrong side
#define RAY_OFFSET 0.001f

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
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

// GPU Gems 3, ch 20, eq 7-9
__host__ __device__ glm::vec3 calculateRandomDirectionInSpecularLobe(
    glm::vec3 direction,
    float exponent,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float cosTheta = powf(u01(rng), 1.0f / (exponent + 1.0f));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = u01(rng) * TWO_PI;

    glm::vec3 notParallel = fabsf(direction.x) < SQRT_OF_ONE_THIRD
        ? glm::vec3(1, 0, 0)
        : (fabsf(direction.y) < SQRT_OF_ONE_THIRD ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
    glm::vec3 tangent = glm::normalize(glm::cross(direction, notParallel));
    glm::vec3 bitangent = glm::cross(direction, tangent);

    return cosTheta * direction
        + cosf(phi) * sinTheta * tangent
        + sinf(phi) * sinTheta * bitangent;
}

__host__ __device__ float schlickFresnel(float cosTheta, float etaI, float etaT)
{
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    float m = 1.0f - cosTheta;
    float m2 = m * m;
    return r0 + (1.0f - r0) * m2 * m2 * m;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    const glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
    glm::vec3 newDirection;
    glm::vec3 attenuation;
    // transmitted rays exit the other side
    float offsetSign = 1.0f;

    if (m.hasRefractive > 0.0f)
    {
        // normal always faces the ray, so only outside tells us the medium
        const float etaI = outside ? 1.0f : m.indexOfRefraction;
        const float etaT = outside ? m.indexOfRefraction : 1.0f;
        const float eta = etaI / etaT;

        const float cosThetaI = fminf(1.0f, fmaxf(0.0f, glm::dot(-incident, normal)));
        glm::vec3 refracted = glm::refract(incident, normal, eta);

        // glm::refract returns 0 on TIR
        const bool totalInternalReflection = glm::length2(refracted) < EPSILON;
        const float reflectProbability = totalInternalReflection
            ? 1.0f
            : schlickFresnel(cosThetaI, etaI, etaT);

        if (u01(rng) < reflectProbability)
        {
            newDirection = glm::reflect(incident, normal);
        }
        else
        {
            newDirection = refracted;
            offsetSign = -1.0f;
        }
        // branch prob == F, so F cancels out
        attenuation = m.specular.color;
    }
    else if (m.hasReflective > 0.0f && u01(rng) < m.hasReflective)
    {
        const glm::vec3 mirror = glm::reflect(incident, normal);
        newDirection = m.specular.exponent > 0.0f
            ? calculateRandomDirectionInSpecularLobe(mirror, m.specular.exponent, rng)
            : mirror;
        attenuation = m.specular.color / m.hasReflective;
    }
    else
    {
        // cosine-weighted, so pdf and BRDF cancel down to one multiply
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        const float diffuseProbability = 1.0f - m.hasReflective;
        attenuation = diffuseProbability > 0.0f ? m.color / diffuseProbability : m.color;
    }

    pathSegment.ray.direction = glm::normalize(newDirection);
    pathSegment.ray.origin = intersect + normal * (RAY_OFFSET * offsetSign);
    pathSegment.color *= attenuation;
    pathSegment.remainingBounces--;
}
