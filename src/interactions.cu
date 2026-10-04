#include "interactions.h"

#include "utilities.h"

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

__host__ __device__
bool refractRay(glm::vec3& rayDir, glm::vec3 n, float eta)
{
    float cosThetaI = glm::dot(-rayDir, n);

    float sin2ThetaI =
        fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);

    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1.0f)
        return false;

    float cosThetaT =
        sqrtf(1.0f - sin2ThetaT);

    rayDir =
        eta * rayDir +
        (eta * cosThetaI - cosThetaT) * n;

    return true;
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect;

}

__host__ __device__
float fresnelDielectric(
    float cosThetaI,
    float etaI,
    float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);

    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 *= r0;

    return r0 + (1.0f - r0) * powf(1.0f - cosThetaI, 5.0f);
}