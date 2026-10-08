#include "bsdf.h"
#include "interactions.h"   
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 sampleDiffuse(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    thrust::default_random_engine& rng)
{
    // wi
    glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
    ray.origin = intersect + 0.001f * wi;
    ray.direction = wi;

    pdf = glm::dot(wi, normal) / PI;

    glm::vec3 bsdf = m.color / PI;
    return bsdf;
}

__host__ __device__ glm::vec3 sampleSpecularReflect(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    thrust::default_random_engine& rng)
{
    // wi
	glm::vec3 wi = glm::reflect(ray.direction, normal);
    glm::vec3 offsetNormal = glm::dot(wi, normal) > 0.0f ? normal : -normal;
    ray.origin = intersect + 0.001f * offsetNormal;
    ray.direction = wi;
    
	pdf = 1.0f;
	glm::vec3 bsdf = m.color / glm::dot(wi, normal);
    return bsdf;
}

__host__ __device__ float computeFresnelDielectric(
    float cosThetaI,
    float etaI,
    float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // exiting 
    if (cosThetaI <= 0.0f) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = fabsf(cosThetaI);
    }

    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // total internal reflection
    if (sinThetaT >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
}

__host__ __device__ glm::vec3 sampleDielectric(
    Ray& ray,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float& pdf,
    float ior,
    thrust::default_random_engine& rng)
{   
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r = u01(rng);

    // try refracting to check TIR case
    glm::vec3 wo = - ray.direction;
    float cosThetaI = glm::dot(wo, normal); // wo, nor

    float Fresnel = computeFresnelDielectric(cosThetaI, 1.0f, ior);
    
    glm::vec3 wi;

    if (r < Fresnel) { // REFLECT if TIR or fresnel is close to 1 OR 
                       //TIR (in this case Fresnel ==1)
        wi = glm::reflect(ray.direction, normal);
        glm::vec3 offsetNormal = glm::dot(wi, normal) > 0.0f ? normal : -normal;
        ray.origin = intersect + 0.001f * offsetNormal;
        ray.direction = wi;
    }
    else { // TRANSMIT if fresnel is close to 0
        float eta;
        glm::vec3 n = normal;
        if (cosThetaI < 0.0f) {
            // exit
            n = -normal;
            eta = ior / 1.0f;
            cosThetaI = -cosThetaI;
        }
        else {
            // enter
            eta = 1.0f / ior;
        }

        wi = glm::refract(-wo, n, eta);

        glm::vec3 offsetNormal = glm::dot(wi, normal) > 0.0f ? normal : -normal;
        ray.origin = intersect + 0.001f * offsetNormal;
		ray.direction = wi;
    }

    pdf = 1.0f;
    float cosTheta = glm::max(glm::abs(glm::dot(wi, normal)), 1e-6f);
    return m.color / cosTheta;
}
