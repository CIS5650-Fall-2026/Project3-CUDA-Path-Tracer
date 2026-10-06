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

/**
* cosThetaI: cosine of angle between incoming ray and surface normal
* eta: indexofRefraction (compared to air), 1.0 doesnt bend light
* return: R (fraction of light that reflects, 1-R passes through and refracts
**/
__host__ __device__ float FrDielectric(float cosThetaI, float eta) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    if (cosThetaI < 0.f) {
        eta = 1.f / eta; // from coming out of the material into the air
        cosThetaI = -cosThetaI; // for coming from the inside of the material, reverse normal so facing inward
    }
    float sin2ThetaI = 1.f - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.f) return 1.f;
    float cosThetaT = sqrtf(1.f - sin2ThetaT);
    float rParl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5f * (rParl * rParl + rPerp * rPerp);
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{


    thrust::uniform_real_distribution<float> u01(0, 1); // rng
    auto diff_col = (m.color.r + m.color.g + m.color.b) / 3.f;
    auto spec_col = (m.specular.color.r + m.specular.color.g + m.specular.color.b) / 3.f;
    auto prob_spec = (m.hasReflective > 0.f && diff_col + spec_col > 0.f) ? spec_col / (diff_col + spec_col) : 0.f;
    // calculate prob_spec based on the colors
    float refract = m.hasRefractive;
    glm::vec3 wi = pathSegment.ray.direction; // initial direction

    glm::vec3 dir;

    if (refract > 0.f) {
        // refract / reflect mix for refracting surfaces
        float cosI = glm::dot(-wi, normal);
        float R = FrDielectric(cosI, m.indexOfRefraction);
        bool entering = cosI > 0.f;
        glm::vec3 n = entering ? normal : -normal;
        float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;

        if (u01(rng) < R) {
            dir = glm::reflect(wi, n);
            pathSegment.color *= m.color;
        }
        else {
            dir = glm::refract(wi, n, eta);
            pathSegment.color *= m.specular.color;
        }
    }
    else {
        // fall back to typical diffuse / reflect mix
        if (prob_spec <= u01(rng)) {
            dir = calculateRandomDirectionInHemisphere(normal, rng);
            pathSegment.color *= m.color / (1.f - prob_spec);
        }
        else {
            // reflect
            dir = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color / prob_spec;
        }
        //pathSegment.color *= m.color;
    }
    pathSegment.ray.origin = intersect + dir * 0.001f;
    pathSegment.ray.direction = dir;
    --pathSegment.remainingBounces;
    if (pathSegment.remainingBounces <= 0) {
        pathSegment.color = glm::vec3(0.f);
    }
}
