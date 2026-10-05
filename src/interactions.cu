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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
    glm::vec3 n = glm::normalize(normal);
    const float OFFSET = 0.001f;

    if (m.hasRefractive > 0.0f)
    {
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        float etaI = 1.0f;
        float etaT = m.indexOfRefraction;

        float cosTheta = glm::dot(-incoming, n);

        if (cosTheta < 0.0f)
        {
            n = -n;

            float temp = etaI;
            etaI = etaT;
            etaT = temp;

            cosTheta = glm::dot(-incoming, n);
        }
        cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);
        float eta = etaI / etaT;

        glm::vec3 refracted = glm::refract(incoming, n, eta);

        bool totalInternalReflection = glm::length(refracted) < 1e-8f;

        float r0 = (etaI - etaT) / (etaI + etaT);
        r0 *= r0;
        float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);

        if (totalInternalReflection || u01(rng) < fresnel)
        {
            pathSegment.ray.direction = glm::reflect(incoming, n);
        }
        else
        {
            pathSegment.ray.direction = glm::normalize(refracted);
        }

        pathSegment.ray.origin = intersect + pathSegment.ray.direction * OFFSET;

        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;

        return;
    }

    if (m.hasReflective > 0.0f)
    {
        pathSegment.ray.direction = glm::normalize(glm::reflect(incoming, n));

        pathSegment.ray.origin = intersect + pathSegment.ray.direction * OFFSET;

        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;

        return;
    }

    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * OFFSET;


    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}
