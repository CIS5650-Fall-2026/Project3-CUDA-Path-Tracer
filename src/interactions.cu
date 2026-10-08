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
    // If this was the last bounce and we HAVEN'T hit a light yet, ray is black!
    pathSegment.remainingBounces--;
    if (pathSegment.remainingBounces == 0) {
        pathSegment.color = glm::vec3(0.f);
        return;
    }
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Accumulate surface color
    pathSegment.color *= m.color;

    // Refractive lobe
    if (m.hasRefractive > 0.f) {
        float eta;
        glm::vec3 n;
        glm::vec3 dir = pathSegment.ray.direction;
        glm::vec3 newDir;

        // Fresnel branch
        if (u01(rng) < fresnelDielectric(glm::dot(-dir, normal), m.indexOfRefraction)) {
            newDir = glm::reflect(dir, normal);
        } 
        // Refraction branch
        else {
            // Flip normal/IOR depending on entering/leaving glass
            if (glm::dot(dir, normal) < 0) {
                n = normal;
                eta = 1.f / m.indexOfRefraction;
            } else {
                n = -normal;
                eta = m.indexOfRefraction;
            }
            newDir = glm::refract(dir, n, eta);
        }

        // Total internal reflection
        if (newDir == glm::vec3(0.f)) {
            newDir = glm::reflect(dir, n);
        }
            
        pathSegment.ray.direction = newDir;
    }
    // Specular lobe
    else if (u01(rng) < m.hasReflective) {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    } 
    // Diffuse lobe
    else {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }

    // Update new path in place
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
}

// https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission
__host__ __device__ float fresnelDielectric(float cosThetaI, float ior) {
    if (cosThetaI < 0) {
        ior = 1 / ior;
        cosThetaI = -cosThetaI;
    }

    // Compute cos theta t using snells law
    float sin2Theta_i = 1 - (cosThetaI*cosThetaI);
    float sin2Theta_t = sin2Theta_i / (ior*ior);
    if (sin2Theta_t >= 1)
        return 1.f;
    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sin2Theta_t));

    float r_parl = (ior * cosThetaI - cosThetaT) /
                   (ior * cosThetaI + cosThetaT);
    float r_perp = (cosThetaI - ior * cosThetaT) /
                   (cosThetaI + ior * cosThetaT);

    return (r_parl*r_parl + r_perp*r_perp) / 2;
}