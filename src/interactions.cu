#include "interactions.h"

#include "utilities.h"
#include "features.h"

#include <thrust/random.h>

#define REFRACTION_OFFSET 0.001f

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

#if REFRACTION
__host__ __device__ void scatterRefractive(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 incident = pathSegment.ray.direction;
    float ior = m.indexOfRefraction;
    float eta = outside ? 1.0f / ior : ior;

    float cosI = glm::min(-glm::dot(incident, normal), 1.0f);
    float sin2T = eta * eta * (1.0f - cosI * cosI);

    float fresnel = 1.0f;
    float cosT = 0.0f;
    if (sin2T < 1.0f) {
        cosT = sqrtf(1.0f - sin2T);
        float r0 = (1.0f - ior) / (1.0f + ior);
        r0 *= r0;
        float c = 1.0f - (outside ? cosI : cosT);
        fresnel = r0 + (1.0f - r0) * c * c * c * c * c;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) < fresnel) {
        pathSegment.ray.direction = glm::normalize(glm::reflect(incident, normal));
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;
    } else {
        pathSegment.ray.direction = glm::normalize(eta * incident + (eta * cosI - cosT) * normal);
        pathSegment.ray.origin = intersect - normal * REFRACTION_OFFSET;
    }
    pathSegment.color *= m.specular.color;
}
#endif

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
#if REFRACTION
    if (m.hasRefractive > 0.0f)
    {
        scatterRefractive(pathSegment, intersect, normal, outside, m, rng);
        pathSegment.remainingBounces--;
        return;
    }
#endif

    glm::vec3 newDirection;

#if ENABLE_SPECULAR
    float probSpecular = m.hasReflective;
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (probSpecular >= 1.0f || (probSpecular > 0.0f && u01(rng) < probSpecular))
    {
        newDirection = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color / probSpecular;
    }
    else
    {
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= (probSpecular > 0.0f)
            ? m.color / (1.0f - probSpecular)
            : m.color;
    }
#else
    newDirection = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.color *= m.color;
#endif

    pathSegment.ray.direction = glm::normalize(newDirection);

    // Offset along the new direction so the ray can't re-hit the surface it
    // just left (shadow acne).
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;
    pathSegment.remainingBounces--;
}
