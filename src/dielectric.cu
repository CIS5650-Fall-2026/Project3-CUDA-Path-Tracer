#include "dielectric.h"

#include "microfacet.h"

namespace Dielectric
{

__device__ float fresnel_dielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = fminf(fmaxf(cosThetaI, -1.0f), 1.0f);
    if (cosThetaI < 0.0f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = -cosThetaI;
    }

    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                      ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5f * (rParallel * rParallel + rPerp * rPerp);
}

__device__ ScatterResult calculate_transmission(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    bool smooth,
    glm::vec3 woWorld,
    const glm::vec3& sampledWm,
    thrust::default_random_engine& rng)
{
    ScatterResult result;
    result.throughputMultiplier = glm::vec3(0.0f);
    result.contribution = glm::vec3(0.0f);
    result.etaScale = 1.0f;
    result.pdf = 0.0f;
    result.wasSpecular = false;
    result.wasTransmission = false;
    glm::vec3 n = smooth ? normal : sampledWm;
    if (glm::dot(woWorld, n) < 0.0f) {
        n = -n;
    }

    glm::vec3 rayDir = glm::normalize(pathSegment.ray.direction);
    float ior = fmaxf(m.indexOfRefraction, 1.0001f);
    float etaI = outside ? 1.0f : ior;
    float etaT = outside ? ior : 1.0f;
    float etaP = etaT / etaI;

    float cosThetaO = fminf(fmaxf(glm::dot(woWorld, n), 0.0f), 1.0f);
    float R = fresnel_dielectric(cosThetaO, etaI, etaT);
    float T = 1.0f - R;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);
    glm::vec3 wi;

    bool sampledReflection = random < R;
    if (sampledReflection) {
        wi = glm::reflect(rayDir, n);
        result.pdf = R;
    } else {
        wi = glm::refract(rayDir, n, etaI / etaT);
        if (glm::dot(wi, wi) <= 0.0f) {
            return result;
        }
        result.pdf = T;
        result.etaScale = etaP * etaP;
        result.wasTransmission = true;
    }

    if (!smooth) {
        glm::vec3 wiWorld = glm::normalize(wi);
        float macroCosThetaI = glm::dot(normal, wiWorld);
        if ((sampledReflection && macroCosThetaI <= 0.0f) ||
            (!sampledReflection && macroCosThetaI >= 0.0f)) {
            return result;
        }

        glm::vec3 bsdf = Microfacet::evaluate_dielectric_bsdf(
            woWorld, wiWorld, normal, m, outside, Microfacet::material_alpha(m));
        float pdf = Microfacet::evaluate_dielectric_pdf(
            woWorld, wiWorld, normal, m, outside, Microfacet::material_alpha(m));
        if (pdf <= 0.0f) {
            return result;
        }

        float cosTheta = fabsf(glm::dot(normal, wiWorld));
        result.throughputMultiplier = bsdf * cosTheta / pdf;
        result.pdf = pdf;
    } else {
        result.throughputMultiplier = m.color;
        if (random >= R) {
            result.throughputMultiplier /= etaP * etaP;
        }
    }

    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.ray.origin =
        intersect + 0.0002f * pathSegment.ray.direction;
    result.wasSpecular = smooth;

    return result;
}

__device__ glm::vec3 get_volume_transmittance(
    const glm::vec3& attenuationColor,
    float attenuationDistance,
    float distance) {
    if (attenuationDistance <= 0.0f || attenuationDistance >= FLT_MAX ||
        distance <= 0.0f) {
        return glm::vec3(1.0f);
    }

    float x = distance / attenuationDistance;
    return glm::pow(attenuationColor, glm::vec3(x));
}

}
