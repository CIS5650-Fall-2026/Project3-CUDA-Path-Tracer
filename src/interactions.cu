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
    const float EPS = 0.001f;
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
    glm::vec3 outdir;

    // flip normal if this is an internal ray
    bool innerNormal = glm::dot(incident, normal) > 0.0f;
    if (innerNormal) {
        normal = -normal;
    }
    glm::vec3 offset = normal;

    if (m.hasRefractive > 0.0f)
    {

        //check range for IOR
        float indexOfRefraction = m.indexOfRefraction;
        if (indexOfRefraction <= 0.0f) {
            indexOfRefraction = 1.0f;
        }
        float eta = 1.0f / indexOfRefraction;
        if (innerNormal) {
            eta = indexOfRefraction;
        }
        //refract using the incident, normal and calculated eta
        glm::vec3 refracted = glm::refract(incident, normal, eta);

        // Calculated the transmitted angle
        float transmitAngle = glm::dot(-incident, normal);
        if (innerNormal && glm::dot(refracted, refracted) > 0.0f) {
            transmitAngle = glm::dot(glm::normalize(refracted), -normal);
        }
        //apply snells law
        float r0 = (1.0f - indexOfRefraction) / (1.0f + indexOfRefraction);
        //square it
        r0 = r0 * r0;
        float fresnel = r0 + (1.0f - r0) * glm::pow(1.0f - transmitAngle, 5.0f);

        // reflect with fresnel probability or if refracted/reflected rays are the same
        if (u01(rng) < fresnel ||glm::dot(refracted, refracted) == 0.0f) {
            outdir = glm::reflect(incident, normal);
        }
        else {
            outdir = glm::normalize(refracted);
            offset = -normal;
        }
    }
    else if (m.hasReflective > 0.0f) {
        // perfect mirror, so just reflect
        outdir = glm::reflect(incident, normal);
    }
    else {
        // Diffuse
        outdir = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.direction = glm::normalize(outdir);
    pathSegment.ray.origin = intersect + offset * EPS;
    pathSegment.color *= m.color;
}
