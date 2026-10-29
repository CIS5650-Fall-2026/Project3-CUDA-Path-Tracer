#include "render/bsdf.h"

#include "render/sampling.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 offsetOrigin(glm::vec3 hitPoint, glm::vec3 normal)
{
    const float scale = fmaxf(1.0f, maxComponent(glm::abs(hitPoint)));
    return hitPoint + normal * (EPSILON * scale);
}

namespace
{
// Below this alpha (roughness 0.01) a lobe is a perfect mirror or clean refraction.
constexpr float MIN_ALPHA = 1e-4f;

// Smith's Lambda for isotropic GGX.
__host__ __device__ float ggxLambda(float cosTheta, float alpha)
{
    // A cosine of exactly 0 would make G2/G1 a NaN
    const float cos2 = fmaxf(cosTheta * cosTheta, 1e-8f);
    const float tan2 = fmaxf(1.0f - cos2, 0.0f) / cos2;
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tan2) - 1.0f);
}

// f * cos / pdf of a visible-normal GGX sample without the Fresnel term:
// height-correlated G2 over G1 (Heitz 2014).
__host__ __device__ float smithG2OverG1(float cosO, float cosI, float alpha)
{
    if (alpha < MIN_ALPHA)
    {
        return 1.0f;
    }
    const float lambdaO = ggxLambda(cosO, alpha);
    return (1.0f + lambdaO) / (1.0f + lambdaO + ggxLambda(cosI, alpha));
}

// Visible-normal GGX sampling (Dupuy and Benyoub 2023, "Sampling Visible GGX Normals with Spherical Caps").
__host__ __device__ glm::vec3 sampleMicrofacetNormal(glm::vec3 n, glm::vec3 wo, float alpha,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float u1 = u01(rng);
    const float u2 = u01(rng);
    if (alpha < MIN_ALPHA)
    {
        return n;
    }
    glm::vec3 t;
    glm::vec3 b;
    frameAround(n, t, b);
    const glm::vec3 woStd = glm::normalize(glm::vec3(alpha * glm::dot(wo, t), alpha * glm::dot(wo, b), glm::dot(wo, n)));

    const float phi = TWO_PI * u1;
    const float z = (1.0f - u2) * (1.0f + woStd.z) - woStd.z;
    const float sinTheta = sqrtf(glm::clamp(1.0f - z * z, 0.0f, 1.0f));
    const glm::vec3 hStd = glm::vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), z) + woStd;

    const glm::vec3 h = glm::normalize(glm::vec3(alpha * hStd.x, alpha * hStd.y, fmaxf(hStd.z, 1e-6f)));
    return h.x * t + h.y * b + h.z * n;
}

// KHR_materials_ior reserves ior = 0 for a Fresnel term of 1 everywhere.
__host__ __device__ float iorF0(float ior)
{
    if (ior == 0.0f)
    {
        return 1.0f;
    }
    const float r = (ior - 1.0f) / (ior + 1.0f);
    return r * r;
}

// Reflected share of a glass microfacet; Schlick takes the cosine on the air side.
__host__ __device__ glm::vec3 glassFresnel(const Material& m, bool outside, float eta, float cosO, glm::vec3 f0)
{
    const float sin2T = eta * eta * (1.0f - cosO * cosO);
    if (m.ior == 0.0f || sin2T >= 1.0f)
    {
        return glm::vec3(1.0f);
    }
    const float cosSchlick = outside ? cosO : sqrtf(1.0f - sin2T);
    return m.specularFactor * schlickFresnel(f0, cosSchlick);
}

// GGX D; sin^2 from the cross product stays accurate near the peak of a smooth lobe.
__host__ __device__ float ggxD(glm::vec3 n, glm::vec3 h, float alpha)
{
    const float cosH = glm::dot(n, h);
    if (cosH <= 0.0f)
    {
        return 0.0f;
    }
    const glm::vec3 c = glm::cross(n, h);
    const float a2 = alpha * alpha;
    const float d = cosH * cosH * a2 + glm::dot(c, c);
    return a2 / (PI * d * d);
}

// f * cos without the Fresnel term, and the visible-normal sampling density.
struct LobeEval
{
    float fCos;
    float pdf;
};

__host__ __device__ LobeEval ggxReflection(glm::vec3 n, glm::vec3 h, float cosO, float cosI, float alpha)
{
    const float d = ggxD(n, h, alpha);
    const float lambdaO = ggxLambda(cosO, alpha);
    const float g2 = 1.0f / (1.0f + lambdaO + ggxLambda(cosI, alpha));
    const float g1 = 1.0f / (1.0f + lambdaO);
    return { d * g2 / (4.0f * cosO), d * g1 / (4.0f * cosO) };
}
}  // namespace

__host__ __device__ BsdfEval evalPbr(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s)
{
    BsdfEval result = { glm::vec3(0.0f), 0.0f };
    const float alpha = s.roughness * s.roughness;
    const bool reflected = glm::dot(wi, surfaceNormal) > 0.0f;

    // scatterPbr's lobe probabilities
    const float pCoat = hasClearcoat(m, outside)
        ? m.clearcoat * schlickFresnel(glm::vec3(0.04f), glm::dot(s.coatNormal, wo)).x : 0.0f;
    const float pMetal = (1.0f - pCoat) * s.metallic;
    const float pGlass = (1.0f - pCoat) * (1.0f - s.metallic) * s.transmission;
    const float pLayered = (1.0f - pCoat) * (1.0f - s.metallic) * (1.0f - s.transmission);

    const float coatAlpha = m.clearcoatRoughness * m.clearcoatRoughness;
    if (pCoat > 0.0f && coatAlpha >= MIN_ALPHA && reflected)
    {
        const glm::vec3 n = s.coatNormal;
        const float cosO = glm::dot(n, wo);
        const float cosI = glm::dot(n, wi);
        if (cosO > 0.0f && cosI > 0.0f)
        {
            const LobeEval coat = ggxReflection(n, glm::normalize(wo + wi), cosO, cosI, coatAlpha);
            result.fCos += glm::vec3(pCoat * coat.fCos);
            result.pdf += pCoat * coat.pdf;
        }
    }

    const glm::vec3 n = s.normal;
    const float cosO = glm::dot(n, wo);
    const float cosI = glm::dot(n, wi);
    if (cosO <= 0.0f)
    {
        return result;
    }
    const glm::vec3 f0 = glm::min(iorF0(m.ior) * m.specularColorFactor, glm::vec3(1.0f));
    const float specular = m.specularFactor;
    const float eta = outside ? 1.0f / m.ior : m.ior;  // incident over transmitted index

    if (reflected && cosI > 0.0f)
    {
        const glm::vec3 h = glm::normalize(wo + wi);
        const float cosOH = glm::dot(wo, h);
        const bool rough = alpha >= MIN_ALPHA;
        const LobeEval ggx = rough ? ggxReflection(n, h, cosO, cosI, alpha) : LobeEval{ 0.0f, 0.0f };

        if (pMetal > 0.0f && rough)
        {
            result.fCos += pMetal * schlickFresnel(s.baseColor, cosOH) * ggx.fCos;
            result.pdf += pMetal * ggx.pdf;
        }
        if (pGlass > 0.0f && rough)
        {
            const glm::vec3 fresnel = glassFresnel(m, outside, eta, cosOH, f0);
            result.fCos += pGlass * fresnel * ggx.fCos;
            result.pdf += pGlass * maxComponent(fresnel) * ggx.pdf;
        }
        if (pLayered > 0.0f)
        {
            const float layerEstimate = specular * maxComponent(schlickFresnel(f0, cosO));
            const float baseEstimate = maxComponent(s.baseColor);
            if (layerEstimate + baseEstimate > 0.0f)
            {
                const float pLayer = layerEstimate / (layerEstimate + baseEstimate);
                const glm::vec3 layer = specular * schlickFresnel(f0, cosOH);
                if (rough)
                {
                    result.fCos += pLayered * layer * ggx.fCos;
                    result.pdf += pLayered * pLayer * ggx.pdf;
                }
                result.fCos += pLayered * s.baseColor * (1.0f - maxComponent(layer)) * (cosI / PI);
                result.pdf += pLayered * (1.0f - pLayer) * (cosI / PI);
            }
        }
    }
    else if (!reflected && cosI < 0.0f && pGlass > 0.0f && alpha >= MIN_ALPHA && m.ior != 1.0f && m.ior != 0.0f)
    {
        // Generalized half vector of the refraction (Walter et al. 2007), turned to wo's side
        glm::vec3 h = glm::normalize(eta * wo + wi);
        if (glm::dot(h, n) < 0.0f)
        {
            h = -h;
        }
        const float cosOH = glm::dot(wo, h);
        const float cosIH = glm::dot(wi, h);
        const glm::vec3 fresnel = glassFresnel(m, outside, eta, cosOH, f0);
        const float pReflect = maxComponent(fresnel);
        if (cosOH > 0.0f && cosIH < 0.0f && pReflect < 1.0f)
        {
            const float d = ggxD(n, h, alpha);
            const float lambdaO = ggxLambda(cosO, alpha);
            const float g2 = 1.0f / (1.0f + lambdaO + ggxLambda(-cosI, alpha));
            const float g1 = 1.0f / (1.0f + lambdaO);
            const float denom = eta * cosOH + cosIH;
            const float jacobian = -cosIH / (denom * denom);
            const float pdf = pGlass * (1.0f - pReflect) * g1 * d * cosOH / cosO * jacobian;
            result.pdf += pdf;
            result.fCos += pdf * (g2 / g1) * s.baseColor;
        }
    }
    return result;
}

__host__ __device__ bool scatterPbr(
    PathSegment& path,
    glm::vec3 hitPoint,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec3 wo = -path.ray.direction;
    const float alpha = s.roughness * s.roughness;

    glm::vec3 wi;
    glm::vec3 lobeNormal;
    glm::vec3 weight;
    bool transmitted = false;
    bool smooth = alpha < MIN_ALPHA;

    // Each lobe is picked with its mixing weight, so the weight cancels;
    // the specular layer's depends on the microfacet, so it uses an estimate from wo.
    if (hasClearcoat(m, outside)
        && u01(rng) < m.clearcoat * schlickFresnel(glm::vec3(0.04f), glm::dot(s.coatNormal, wo)).x)
    {
        const float coatAlpha = m.clearcoatRoughness * m.clearcoatRoughness;
        lobeNormal = s.coatNormal;
        const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, coatAlpha, rng);
        wi = glm::reflect(-wo, h);
        weight = glm::vec3(smithG2OverG1(glm::dot(lobeNormal, wo), fabsf(glm::dot(lobeNormal, wi)), coatAlpha));
        smooth = coatAlpha < MIN_ALPHA;
    }
    else if (u01(rng) < s.metallic)
    {
        lobeNormal = s.normal;
        const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, alpha, rng);
        wi = glm::reflect(-wo, h);
        weight = schlickFresnel(s.baseColor, glm::dot(wo, h))
               * smithG2OverG1(glm::dot(lobeNormal, wo), fabsf(glm::dot(lobeNormal, wi)), alpha);
    }
    else
    {
        // Dielectric: KHR_materials_specular scales the Fresnel term and f0
        lobeNormal = s.normal;
        const glm::vec3 f0 = glm::min(iorF0(m.ior) * m.specularColorFactor, glm::vec3(1.0f));
        const float specular = m.specularFactor;

        if (u01(rng) < s.transmission)
        {
            // Glass: reflect or refract through one microfacet (Walter et al. 2007)
            const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, alpha, rng);
            const float eta = outside ? 1.0f / m.ior : m.ior;  // incident over transmitted index
            const glm::vec3 fresnel = glassFresnel(m, outside, eta, glm::dot(wo, h), f0);
            const float pReflect = maxComponent(fresnel);
            const float cosWo = glm::dot(lobeNormal, wo);
            if (u01(rng) < pReflect)
            {
                wi = glm::reflect(-wo, h);
                weight = fresnel / pReflect * smithG2OverG1(cosWo, fabsf(glm::dot(lobeNormal, wi)), alpha);
            }
            else
            {
                wi = glm::refract(-wo, h, eta);
                weight = s.baseColor * smithG2OverG1(cosWo, fabsf(glm::dot(lobeNormal, wi)), alpha);
                transmitted = true;
                smooth = smooth || m.ior == 1.0f;
            }
        }
        else
        {
            // Specular layer over Lambert. The base estimate is not scaled by
            // what the layer takes: at grazing wo that would turn rare
            // diffuse samples into fireflies.
            const float cosWo = glm::dot(lobeNormal, wo);
            const float layerEstimate = specular * maxComponent(schlickFresnel(f0, cosWo));
            const float baseEstimate = maxComponent(s.baseColor);
            if (layerEstimate + baseEstimate <= 0.0f)
            {
                return false;
            }
            const float pLayer = layerEstimate / (layerEstimate + baseEstimate);
            if (u01(rng) < pLayer)
            {
                const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, alpha, rng);
                wi = glm::reflect(-wo, h);
                weight = specular * schlickFresnel(f0, glm::dot(wo, h)) / pLayer
                       * smithG2OverG1(cosWo, fabsf(glm::dot(lobeNormal, wi)), alpha);
            }
            else
            {
                wi = sampleCosineHemisphere(lobeNormal, rng);
                const glm::vec3 h = glm::normalize(wo + wi);
                const float layer = specular * maxComponent(schlickFresnel(f0, glm::dot(wo, h)));
                weight = s.baseColor * (1.0f - layer) / (1.0f - pLayer);
                smooth = false;
            }
        }
    }

    // A reflection leaves on the lit side of both normals, a transmission on the far side of both
    const float side = transmitted ? -1.0f : 1.0f;
    if (side * glm::dot(wi, lobeNormal) <= 0.0f || side * glm::dot(wi, surfaceNormal) <= 0.0f)
    {
        return false;
    }

    path.ray.direction = glm::normalize(wi);
    path.ray.origin = offsetOrigin(hitPoint, side * surfaceNormal);
    path.color *= weight;
    // The density over every rough lobe, the same one NEE weighs against
    path.pdf = smooth ? 0.0f : evalPbr(wo, path.ray.direction, surfaceNormal, outside, m, s).pdf;
    return true;
}
