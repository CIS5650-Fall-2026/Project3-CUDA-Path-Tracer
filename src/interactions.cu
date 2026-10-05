#include "interactions.h"

#include "utilities.h"

#include <cmath>
#include <thrust/random.h>

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

__host__ __device__ glm::vec3 calculateRandomDirectionInPhongLobe(
    glm::vec3 reflectionDirection,
    float exponent,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = pow(u01(rng), 1.0f / (exponent + 1.0f));
    float over = sqrt(max(0.0f, 1.0f - up * up));
    float around = u01(rng) * TWO_PI;

    glm::vec3 directionNotReflection;
    if (abs(reflectionDirection.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotReflection = glm::vec3(1, 0, 0);
    }
    else if (abs(reflectionDirection.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotReflection = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotReflection = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(reflectionDirection, directionNotReflection));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(reflectionDirection, perpendicularDirection1));

    return up * reflectionDirection
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 n = glm::normalize(normal);
    glm::vec3 outDirection;

    if (m.type == MATERIAL_SPECULAR)  // this uses phong lobe to mimic roughness effect
    {
        glm::vec3 reflectionDirection = glm::reflect(glm::normalize(pathSegment.ray.direction), n);
        float roughness = glm::clamp(m.specular.roughness, 0.0f, 1.0f);

        if (roughness <= 0.001f)
        {
            outDirection = reflectionDirection;
        }
        else
        {
            float exponent = max(0.0f, 2.0f / (roughness * roughness) - 2.0f);
            outDirection = calculateRandomDirectionInPhongLobe(reflectionDirection, exponent, rng);

            float cosSurface = glm::dot(n, outDirection);
            if (cosSurface <= 0.0f)
            {
                pathSegment.throughput = glm::vec3(0.0f);
                outDirection = reflectionDirection;
            }
            else
            {
                pathSegment.throughput *= ((exponent + 2.0f) / (exponent + 1.0f)) * cosSurface;
            }
        }
    }
    else if (m.type == MATERIAL_REFRACTIVE)
    {
        glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        float etaI = pathSegment.currentIor;
        float etaT = outside ? m.indexOfRefraction : 1.0f;
        float eta = etaI / etaT;
        float cosTheta = min(glm::dot(-incident, n), 1.0f);
        float sinThetaSquared = max(0.0f, 1.0f - cosTheta * cosTheta);
        bool cannotRefract = eta * eta * sinThetaSquared > 1.0f;

        float r0 = (etaI - etaT) / (etaI + etaT);
        r0 *= r0;
        float reflectance = r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5.0f);
        thrust::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
        if (cannotRefract || uniform01(rng) < reflectance)
        {
            outDirection = glm::reflect(incident, n);
        }
        else
        {
            outDirection = glm::refract(incident, n, eta);
            pathSegment.throughput *= eta * eta;
            pathSegment.currentIor = etaT;
        }
    }
    else
    {
        outDirection = calculateRandomDirectionInHemisphere(n, rng);
    }
    outDirection = glm::normalize(outDirection);

    pathSegment.throughput *= m.color;

    float offsetSign = glm::dot(outDirection, n) >= 0.0f ? 1.0f : -1.0f;  // for refraction
    pathSegment.ray.origin = intersect + offsetSign * (10.0f * EPSILON) * n;  // epsilon is a little too small
    pathSegment.ray.direction = outDirection;
}
