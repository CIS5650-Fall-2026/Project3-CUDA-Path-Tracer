#include "microfacet.h"

#include "utilities.h"

#include <thrust/random.h>

#define REMAP_ROUGHNESS 1

namespace Microfacet
{

__device__ float material_alpha(const Material& m) {
    float alpha = fminf(fmaxf(m.roughness_factor, 0.001f), 1.0f);
#if REMAP_ROUGHNESS
    alpha = alpha * alpha;
#endif
    return alpha;
}

__device__ void material_lobe_weights(const Material& m,
                                      float& wDiffuse,
                                      float& wMetal,
                                      float& wTransmission) {
    wMetal = m.metalic_factor;
    wTransmission = (1.0f - wMetal) * m.transmission_factor;
    wDiffuse = (1.0f - wMetal) * (1.0f - m.transmission_factor);
}

__host__ __device__ float abs_cos_theta(const glm::vec3 &w) {
    return fabsf(w.z);
}

__host__ __device__ bool same_hemisphere(const glm::vec3 &a,
                                         const glm::vec3 &b) {
    return a.z * b.z > 0.0f;
}

__host__ __device__ glm::vec3 to_local(const glm::vec3 &v,
                                       const glm::vec3 &tangent,
                                       const glm::vec3 &bitangent,
                                       const glm::vec3 &normal) {
    return glm::vec3(glm::dot(v, tangent), glm::dot(v, bitangent),
                     glm::dot(v, normal));
}

__host__ __device__ glm::vec3 to_world(const glm::vec3 &v,
                                       const glm::vec3 &tangent,
                                       const glm::vec3 &bitangent,
                                       const glm::vec3 &normal) {
    return v.x * tangent + v.y * bitangent + v.z * normal;
}

__host__ __device__ void make_basis(const glm::vec3 &normal,
                                    glm::vec3 &tangent,
                                    glm::vec3 &bitangent) {
    glm::vec3 helper = fabsf(normal.x) < SQRT_OF_ONE_THIRD
                           ? glm::vec3(1.0f, 0.0f, 0.0f)
                           : glm::vec3(0.0f, 1.0f, 0.0f);
    tangent = glm::normalize(glm::cross(helper, normal));
    bitangent = glm::cross(normal, tangent);
}

__host__ __device__ float get_d_ggx(const glm::vec3 &m, float alpha) {
    float cosTheta = abs_cos_theta(m);
    if (cosTheta <= 0.0f) {
        return 0.0f;
    }

    float alpha2 = alpha * alpha;
    float cosTheta2 = cosTheta * cosTheta;
    float inner = cosTheta2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * inner * inner);
}

__host__ __device__ float get_pdf(const glm::vec3 &wo,
                                  const glm::vec3 &wm,
                                  float alpha) {
    float woDotWm = fabsf(glm::dot(wo, wm));
    if (woDotWm <= 0.0f) {
        return 0.0f;
    }
    return get_d_ggx(wm, alpha) * abs_cos_theta(wm) / (4.0f * woDotWm);
}

__host__ __device__ float get_g_lambda(const glm::vec3 &w, float alpha) {
    float cosTheta = abs_cos_theta(w);
    if (cosTheta <= 0.0f) {
        return 0.0f;
    }

    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta2 = fmaxf(0.0f, 1.0f - cosTheta2);
    float tanTheta2 = sinTheta2 / cosTheta2;
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tanTheta2) - 1.0f);
}

__host__ __device__ float get_g1(const glm::vec3 &w, float alpha) {
    return 1.0f / (1.0f + get_g_lambda(w, alpha));
}

__host__ __device__ float get_g_ggx(const glm::vec3 &wi,
                                    const glm::vec3 &wo,
                                    float alpha) {
    return 1.0f /
           (1.0f + get_g_lambda(wo, alpha) + get_g_lambda(wi, alpha));
}

__host__ __device__ glm::vec3 get_f_ggx(const glm::vec3 &wi,
                                        const glm::vec3 &wo,
                                        const glm::vec3 &f0) {
    glm::vec3 wm = glm::normalize(wi + wo);
    float cosTheta = fminf(fmaxf(fabsf(glm::dot(wi, wm)), 0.0f), 1.0f);
    float oneMinusCos = 1.0f - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    return f0 + (glm::vec3(1.0f) - f0) * oneMinusCos2 * oneMinusCos2 * oneMinusCos;
}

__host__ __device__ float fresnel_dielectric_relative(float cosThetaI,
                                                      float eta) {
    cosThetaI = fminf(fmaxf(cosThetaI, -1.0f), 1.0f);
    float etaI = 1.0f;
    float etaT = eta;
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

__host__ __device__ glm::vec3 sample_wm(float alpha, const glm::vec2 &u) {
    float phi = TWO_PI * u.x;
    float alpha2 = alpha * alpha;
    float cosTheta =
        sqrtf((1.0f - u.y) / (1.0f + (alpha2 - 1.0f) * u.y));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    return glm::normalize(glm::vec3(cosf(phi) * sinTheta,
                                    sinf(phi) * sinTheta,
                                    cosTheta));
}

__host__ __device__ glm::vec3 sample_wm(float alpha,
                                        const glm::vec3& normal,
                                        const glm::vec2& u) {
    glm::vec3 tangent;
    glm::vec3 bitangent;
    make_basis(normal, tangent, bitangent);
    return glm::normalize(to_world(sample_wm(alpha, u), tangent, bitangent,
                                   normal));
}

__host__ __device__ glm::vec3 evaluate_brdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& m,
    float alpha)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    make_basis(normal, tangent, bitangent);

    glm::vec3 wo = to_local(glm::normalize(woWorld), tangent, bitangent,
                            normal);
    glm::vec3 wi = to_local(glm::normalize(wiWorld), tangent, bitangent,
                            normal);
    if (wo.z <= 0.0f || wi.z <= 0.0f || !same_hemisphere(wo, wi))
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 wm = glm::normalize(wi + wo);
    if (wm.z <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    float cosThetaO = abs_cos_theta(wo);
    float cosThetaI = abs_cos_theta(wi);
    float d = get_d_ggx(wm, alpha);
    float g = get_g_ggx(wi, wo, alpha);
    glm::vec3 f = get_f_ggx(wi, wo, m.color);
    return d * g * f / (4.0f * cosThetaI * cosThetaO);
}

__host__ __device__ float evaluate_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    float alpha)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    make_basis(normal, tangent, bitangent);

    glm::vec3 wo = to_local(glm::normalize(woWorld), tangent, bitangent,
                            normal);
    glm::vec3 wi = to_local(glm::normalize(wiWorld), tangent, bitangent,
                            normal);
    if (wo.z <= 0.0f || wi.z <= 0.0f || !same_hemisphere(wo, wi))
    {
        return 0.0f;
    }

    glm::vec3 wm = glm::normalize(wi + wo);
    if (wm.z <= 0.0f)
    {
        return 0.0f;
    }

    return get_pdf(wo, wm, alpha);
}

__host__ __device__ ScatterResult get_brdf_result(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float alpha,
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

        glm::vec3 tangent;
        glm::vec3 bitangent;
        make_basis(normal, tangent, bitangent);

        glm::vec3 woWorld = glm::normalize(-pathSegment.ray.direction);
        glm::vec3 wo = to_local(woWorld, tangent, bitangent, normal);
        if (wo.z <= 0.0f) {
            pathSegment.ray.direction =
                glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect;
            result.throughputMultiplier = m.color;
            result.pdf = 1.0f;
            result.wasSpecular = true;
            return result;
        }

        glm::vec3 wm = to_local(glm::normalize(sampledWm), tangent, bitangent,
                                normal);
        if (wm.z <= 0.0f) {
            wm = -wm;
        }
        glm::vec3 wi = glm::reflect(-wo, wm);
        if (!same_hemisphere(wo, wi) || wi.z <= 0.0f) {
            pathSegment.ray.direction =
                calculateRandomDirectionInHemisphere(normal, rng);
            pathSegment.ray.origin = intersect;
            return result;
        }

        pathSegment.ray.direction =
            glm::normalize(to_world(wi, tangent, bitangent, normal));
        pathSegment.ray.origin = intersect;

        float pdf = get_pdf(wo, wm, alpha);
        if (pdf <= 0.0f) {
            return result;
        }

        float cosThetaI = abs_cos_theta(wi);
        glm::vec3 brdf =
            evaluate_brdf(woWorld, pathSegment.ray.direction, normal, m,
                          alpha);

        result.throughputMultiplier = brdf * cosThetaI / pdf;
        result.pdf = pdf;
        return result;
}

__device__ ScatterResult calculate_reflection(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    bool pureDeltaMetal,
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

    if (pureDeltaMetal) {
        pathSegment.ray.direction =
            glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect;
        result.throughputMultiplier = m.color;
        result.pdf = 1.0f;
        result.wasSpecular = true;
        return result;
    }

    float alpha = m.roughness_factor * m.roughness_factor;
    return get_brdf_result(pathSegment, intersect, normal, m, alpha, sampledWm,
                           rng);
}

__host__ __device__ ScatterResult get_brdf_result(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float alpha,
    thrust::default_random_engine& rng)
{
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec2 u(u01(rng), u01(rng));
        glm::vec3 sampledWm = sample_wm(alpha, normal, u);
        return get_brdf_result(pathSegment, intersect, normal, m, alpha,
                               sampledWm, rng);
}

__host__ __device__ float wm_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wmWorld,
    const glm::vec3& normal,
    float alpha)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    make_basis(normal, tangent, bitangent);
    glm::vec3 wm = to_local(glm::normalize(wmWorld), tangent, bitangent,
                            normal);
    if (wm.z <= 0.0f) {
        wm = -wm;
    }
    return get_d_ggx(wm, alpha) * abs_cos_theta(wm);
}

__host__ __device__ glm::vec3 evaluate_dielectric_bsdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& m,
    bool outside,
    float alpha)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    make_basis(normal, tangent, bitangent);

    glm::vec3 wo = to_local(glm::normalize(woWorld), tangent, bitangent,
                            normal);
    glm::vec3 wi = to_local(glm::normalize(wiWorld), tangent, bitangent,
                            normal);

    float cosThetaO = wo.z;
    float cosThetaI = wi.z;
    if (cosThetaO == 0.0f || cosThetaI == 0.0f) {
        return glm::vec3(0.0f);
    }

    bool reflect = cosThetaI * cosThetaO > 0.0f;
    float eta = fmaxf(m.indexOfRefraction, 1.0001f);
    float etaRel = outside ? eta : 1.0f / eta;
    float etap = reflect ? 1.0f : etaRel;
    glm::vec3 wmUnnormalized = reflect ? wi + wo : wi * etap + wo;
    if (glm::dot(wmUnnormalized, wmUnnormalized) <= 0.0f) {
        return glm::vec3(0.0f);
    }
    glm::vec3 wm = glm::normalize(wmUnnormalized);
    if (wm.z < 0.0f) {
        wm = -wm;
    }

    if (glm::dot(wm, wi) * cosThetaI < 0.0f ||
        glm::dot(wm, wo) * cosThetaO < 0.0f) {
        return glm::vec3(0.0f);
    }

    float F = fresnel_dielectric_relative(glm::dot(wo, wm), etaRel);
    float d = get_d_ggx(wm, alpha);
    float g = get_g_ggx(wi, wo, alpha);

    if (reflect) {
        float denom = fabsf(4.0f * cosThetaI * cosThetaO);
        if (denom <= 0.0f) {
            return glm::vec3(0.0f);
        }
        return m.color * (d * g * F / denom);
    }

    float denom = glm::dot(wi, wm) + glm::dot(wo, wm) / etap;
    denom = denom * denom * cosThetaI * cosThetaO;
    if (denom == 0.0f) {
        return glm::vec3(0.0f);
    }

    float ft = d * (1.0f - F) * g *
               fabsf(glm::dot(wi, wm) * glm::dot(wo, wm) / denom);
    ft /= etap * etap;
    return m.color * ft;
}

__host__ __device__ float evaluate_dielectric_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& m,
    bool outside,
    float alpha)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    make_basis(normal, tangent, bitangent);

    glm::vec3 wo = to_local(glm::normalize(woWorld), tangent, bitangent,
                            normal);
    glm::vec3 wi = to_local(glm::normalize(wiWorld), tangent, bitangent,
                            normal);

    float cosThetaO = wo.z;
    float cosThetaI = wi.z;
    if (cosThetaO == 0.0f || cosThetaI == 0.0f) {
        return 0.0f;
    }

    bool reflect = cosThetaI * cosThetaO > 0.0f;
    float eta = fmaxf(m.indexOfRefraction, 1.0001f);
    float etaRel = outside ? eta : 1.0f / eta;
    float etap = reflect ? 1.0f : etaRel;
    glm::vec3 wmUnnormalized = reflect ? wi + wo : wi * etap + wo;
    if (glm::dot(wmUnnormalized, wmUnnormalized) <= 0.0f) {
        return 0.0f;
    }
    glm::vec3 wm = glm::normalize(wmUnnormalized);
    if (wm.z < 0.0f) {
        wm = -wm;
    }

    if (glm::dot(wm, wi) * cosThetaI < 0.0f ||
        glm::dot(wm, wo) * cosThetaO < 0.0f) {
        return 0.0f;
    }

    float R = fresnel_dielectric_relative(glm::dot(wo, wm), etaRel);
    float T = 1.0f - R;
    float pr = R;
    float pt = T;
    float probSum = pr + pt;
    if (probSum <= 0.0f) {
        return 0.0f;
    }

    float pdfWm = get_d_ggx(wm, alpha) * abs_cos_theta(wm);
    if (reflect) {
        float woDotWm = fabsf(glm::dot(wo, wm));
        if (woDotWm <= 0.0f) {
            return 0.0f;
        }
        return pdfWm / (4.0f * woDotWm) * pr / probSum;
    }

    float denom = glm::dot(wi, wm) + glm::dot(wo, wm) / etap;
    denom *= denom;
    if (denom <= 0.0f) {
        return 0.0f;
    }
    float dwmDwi = fabsf(glm::dot(wi, wm)) / denom;
    return pdfWm * dwmDwi * pt / probSum;
}

}
