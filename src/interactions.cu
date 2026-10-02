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
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 direction;
    pathSegment.specularBounce = false;
    if (m.hasRefractive > 0.0f)
    {
        float ior = glm::max(1.0f, m.indexOfRefraction);
        float eta = outside ? 1.0f / ior : ior;
        float cosine = glm::clamp(-glm::dot(pathSegment.ray.direction, normal), 0.0f, 1.0f);
        float r0 = (1.0f - ior) / (1.0f + ior);
        r0 *= r0;
        float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
        glm::vec3 transmitted = glm::refract(pathSegment.ray.direction, normal, eta);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        if (glm::dot(transmitted, transmitted) < 1e-12f || u01(rng) < fresnel)
        {
            direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else
        {
            direction = transmitted;
            pathSegment.color *= eta * eta;
        }
        pathSegment.specularBounce = true;
    }
    else if (m.hasReflective > 0.0f)
    {
        direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.specularBounce = true;
    }
    else
    {
        direction = calculateRandomDirectionInHemisphere(normal, rng);
    }
    pathSegment.ray.direction = glm::normalize(direction);
    float side = glm::dot(pathSegment.ray.direction, normal) >= 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + normal * (side * 0.004f);
    pathSegment.color *= m.color;
}
