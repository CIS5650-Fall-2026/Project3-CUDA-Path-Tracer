#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    const float u0 = u01(rng);
    const float u1 = u01(rng);
    return cosineSampleHemisphere(normal, glm::vec2(u0, u1));
}

__host__ __device__ glm::vec3 cosineSampleHemisphere(
    glm::vec3 normal,
    glm::vec2 u)
{
    float up = sqrt(u.x); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u.y * TWO_PI;

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

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    const glm::vec2* diffuseSample)
{
    const glm::vec3 incoming =
        glm::normalize(pathSegment.ray.direction);
    const glm::vec3 outwardNormal = glm::normalize(normal);

    const bool entering =
        glm::dot(incoming, outwardNormal) < 0.0f;

    const glm::vec3 facingNormal =
        entering ? outwardNormal : -outwardNormal;

    glm::vec3 outgoing;

    if (m.hasRefractive > 0.0f)
    {
        const float etaI = entering ? 1.0f : m.indexOfRefraction;
        const float etaT = entering ? m.indexOfRefraction : 1.0f;
        const float eta = etaI / etaT;

        const float cosI = glm::clamp(
            -glm::dot(incoming, facingNormal), 0.0f, 1.0f);

        const float sinT2 = eta * eta * (1.0f - cosI * cosI);

        if (sinT2 >= 1.0f)
        {
            // Total internal reflection.
            outgoing = glm::reflect(incoming, facingNormal);
        }
        else
        {
            const float cosT = sqrtf(1.0f - sinT2);

            const float rs =
                (etaI * cosI - etaT * cosT) /
                (etaI * cosI + etaT * cosT);

            const float rp =
                (etaT * cosI - etaI * cosT) /
                (etaT * cosI + etaI * cosT);

            const float reflectance = glm::clamp(
                0.5f * (rs * rs + rp * rp), 0.0f, 1.0f);

            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

            if (u01(rng) < reflectance)
            {
                outgoing = glm::reflect(incoming, facingNormal);
            }
            else
            {
                outgoing =
                    eta * incoming +
                    (eta * cosI - cosT) * facingNormal;

                // Radiance scaling when crossing the interface.
                pathSegment.color *= eta * eta;
            }
        }
    }
    else if (m.hasReflective > 0.0f)
    {
        outgoing = glm::reflect(incoming, facingNormal);
    }
    else if (diffuseSample != nullptr)
    {
        outgoing = cosineSampleHemisphere(facingNormal, *diffuseSample);
    }
    else
    {
        outgoing = calculateRandomDirectionInHemisphere(
            facingNormal, rng);
    }

    outgoing = glm::normalize(outgoing);

    const float offsetSide =
        glm::dot(outgoing, outwardNormal) >= 0.0f ? 1.0f : -1.0f;

    pathSegment.ray.origin =
        intersect + offsetSide * 0.001f * outwardNormal;
    pathSegment.ray.direction = outgoing;

    pathSegment.color *= m.color;
    --pathSegment.remainingBounces;
}