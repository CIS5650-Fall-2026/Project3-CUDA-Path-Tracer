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

__host__ __device__ glm::vec2 calculateRandomPositionOnDisk(thrust::default_random_engine &rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
    xi = xi*2.0f-1.0f;
    if(xi.x == 0 && xi.y == 0)
        return glm::vec2(0.0f);

    glm::vec2 polar;
    if(abs(xi.y) > abs(xi.x)) {
        polar.x = xi.y;
        polar.y = PI*0.5 - PI*0.25 * xi.x/xi.y;
    } else {
        polar.x = xi.x;
        polar.y = PI*0.25 * xi.y/xi.x;
    }
    return polar.x * glm::vec2(cos(polar.y), sin(polar.y));

}

__host__ __device__ glm::vec2 calculateRandomPositionOnSquare(thrust::default_random_engine &rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
    return xi-glm::vec2(0.5f);
}

__host__ __device__ float powerHeuristic(float myPdf, float otherPdf) {
    return myPdf * myPdf / (myPdf*myPdf + otherPdf*otherPdf);
}

__host__ __device__ float pdfBSDF(glm::vec3 wi, glm::vec3 normal, const Material& m, glm::vec3& fLambert) {
    if(m.hasReflective) {
        fLambert = glm::vec3(0.0f);
        return 0.0f;
    } else {
        glm::vec3 bsdf = m.color / PI;
        float cosTheta = abs(glm::dot(wi, normal));
        fLambert = bsdf * cosTheta;
        return dot(wi, normal)*IPI;
    }
}
__host__ __device__ float bsdfF(glm::vec3 wi, glm::vec3 normal, const Material& m) {
    if (m.hasReflective) {
        return 0.0f;
    }
    else {
        return dot(wi, normal) * IPI;
    }
}
__host__ __device__ float pdfLight(glm::vec3 intersect, glm::vec3 lightSamplePos, glm::vec3 lightSampleNorm, int num_lights, const Light& light) {
    glm::vec3 diff = lightSamplePos - intersect;
    glm::vec3 wi = normalize(diff);
    return 1.0f / static_cast<float>(num_lights) * light.localPdf * dot(diff, diff) / abs(dot(lightSampleNorm, -wi));
}

__host__ __device__ void sampleBSDF(
    glm::vec3 wo,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    float& outPdf, glm::vec3& fLambert, glm::vec3& wi)
{
    if(m.hasReflective) {
        // Just use perfect reflection
        wi = -wo - normal * 2.0f * glm::dot(-wo, normal);
        outPdf = 1.0f;

        fLambert = m.color;
    } else {
        wi = calculateRandomDirectionInHemisphere(normal, rng);
        outPdf = dot(wi, normal)*IPI;

        glm::vec3 bsdf = m.color / PI;
        float cosTheta = abs(glm::dot(wi, normal));
        fLambert = bsdf * cosTheta;
    }
}
