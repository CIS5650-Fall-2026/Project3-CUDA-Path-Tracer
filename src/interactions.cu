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

__host__ __device__ float schlickApproximation(float n1, float n2, float cosTheta) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float i = 1 - cosTheta;
    return r0 + (1 - r0) * i * i * i * i * i;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // simplified implementation of refraction using schlick's approximation
    if (m.hasRefractive > 0) {
        thrust::uniform_real_distribution<float> u01(0, 1);

        glm::vec3 incident_norm = glm::normalize(pathSegment.ray.direction);
        float n1 = pathSegment.n;
        float n2 = n1 == 1 ? m.indexOfRefraction : 1;
        float cosTheta = glm::min(glm::dot(-incident_norm, normal), 1.0f);
        float reflectance = schlickApproximation(n1, n2, cosTheta);

        float relRefractiveIdx = n1 / n2;
        float sin2Theta = glm::max(0.0f, 1.0f - cosTheta * cosTheta);
        float sin2Theta_t = relRefractiveIdx * relRefractiveIdx * sin2Theta;
        bool totalInternalReflection = sin2Theta_t >= 1;
        

        if (totalInternalReflection || u01(rng) < reflectance) {
            pathSegment.ray.direction = glm::reflect(incident_norm, normal);
        } else {
            pathSegment.ray.direction = glm::refract(incident_norm, normal, relRefractiveIdx);
            pathSegment.n = n2;
        }
        pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + 0.001f * incident_norm;
        return;
    }
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect + 0.001f * normal;
}
