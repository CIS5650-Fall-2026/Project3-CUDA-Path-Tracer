#include "interactions.h"

#include "utilities.h"

#include "intersections.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    unsigned long long& rng)
{

    float up = sqrt(betterRandom01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = betterRandom01(rng) * TWO_PI;

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
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));

    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    unsigned long long& rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.emittance > 0.0f) {
        pathSegment.color *= m.color * m.emittance;
        pathSegment.remainingBounces = 0;
    } 
    else if(m.hasRefractive > 0){

        bool enterExitFlag = glm::dot(pathSegment.ray.direction, normal) < 0.0f;

        glm::vec3 enterExitNormal = enterExitFlag ? normal : -normal; // to see if we are entering or exiting the object
        float enterEta = enterExitFlag ? 1.0f : m.indexOfRefraction;
        float exitEta = enterExitFlag ? m.indexOfRefraction : 1.0f; // we need to flip it for exit

        float boundaryEta = enterEta / exitEta;

        float reflectionCoefficient = (enterEta - exitEta) / (enterEta + exitEta);
        float baseReflectivity = reflectionCoefficient * reflectionCoefficient;

        float enterCos = glm::min(1.0f, glm::max(0.0f, -glm::dot(pathSegment.ray.direction, enterExitNormal)));
        float fresnelReflectance = baseReflectivity + (1.0f - baseReflectivity) * powf(1.0f - enterCos, 5.0f);

        glm::vec3 refracted = glm::refract(pathSegment.ray.direction, enterExitNormal, boundaryEta);
        bool cannotRefract = glm::length(refracted) < 1e-5f; // just something very little to see if it is basically zero

        if (cannotRefract || betterRandom01(rng) < fresnelReflectance) { // we check if it is even possible and if the probability was also on our side
            pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, enterExitNormal));
            pathSegment.ray.origin = intersect + enterExitNormal * 0.001f;
        } else {
            pathSegment.ray.direction = glm::normalize(refracted);
            pathSegment.ray.origin = intersect - enterExitNormal * 0.001f;
        }
        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;
    
    }   
    else{
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
        pathSegment.remainingBounces--;
    }
}
