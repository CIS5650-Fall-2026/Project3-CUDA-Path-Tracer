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
    normal = glm::normalize(normal);
    glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
    bool entering = glm::dot(incoming, normal) < 0.0f;
    glm::vec3 facingNormal = entering ? normal : -normal;
    glm::vec3 outgoing;

    if (m.hasRefractive) {
        // These materials describe a solid object surrounded by air.
        float incidentIOR = entering ? 1.0f : m.indexOfRefraction;
        float transmittedIOR = entering ? m.indexOfRefraction : 1.0f;
        float ratio = incidentIOR / transmittedIOR;
        float cosine = glm::clamp(-glm::dot(incoming, facingNormal), 0.0f, 1.0f);
        float transmittedSineSquared = ratio * ratio * (1.0f - cosine * cosine);

        float reflectance = 1.0f;
        if (transmittedSineSquared < 1.0f) {
            float base = (incidentIOR - transmittedIOR) / (incidentIOR + transmittedIOR);
            base *= base;
            // Use the transmitted angle when leaving the denser medium.
            float fresnelCosine = incidentIOR > transmittedIOR
                ? sqrtf(1.0f - transmittedSineSquared) : cosine;
            reflectance = base + (1.0f - base) * powf(1.0f - fresnelCosine, 5.0f);
            if (incidentIOR == transmittedIOR) {
                reflectance = 0.0f;
            }
        }

        thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        if (transmittedSineSquared >= 1.0f || uniform(rng) < reflectance) {
            outgoing = glm::reflect(incoming, facingNormal);
        } else {
            outgoing = glm::refract(incoming, facingNormal, ratio);
            // Camera paths carry radiance; entry and exit factors cancel.
            pathSegment.color *= ratio * ratio;
        }
    } else if (m.hasReflective) {
        outgoing = glm::reflect(incoming, facingNormal);
    } else {
        outgoing = calculateRandomDirectionInHemisphere(facingNormal, rng);
    }

    pathSegment.ray.direction = glm::normalize(outgoing);
    float side = glm::dot(outgoing, normal) >= 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + side * normal * 0.0001f;
    // Sampling by Fresnel probability (or cosine for diffuse) leaves just the tint.
    pathSegment.color *= m.color;
}
