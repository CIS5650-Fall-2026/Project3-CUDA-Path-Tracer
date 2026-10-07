#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <cmath>
#include <cfloat>

__host__ __device__ glm::vec3 sampleCosineWeightedHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrtf(u01(rng)); // cos(theta)
    float over = sqrtf(fmaxf(0.0f, 1.0f - up * up)); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (fabsf(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (fabsf(normal.y) < SQRT_OF_ONE_THIRD)
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
        + cosf(around) * over * perpendicularDirection1
        + sinf(around) * over * perpendicularDirection2;
}

static __host__ __device__ void TrowbridgeReitzSample11(
    float cosTheta,
    float U1,
    float U2,
    float& slope_x,
    float& slope_y)
{
    // Special case: normal incidence.
    if (cosTheta > 0.9999f) {
        float r = sqrtf(U1 / (1.0f - U1));
        float phi = 6.28318530718f * U2;

        slope_x = r * cosf(phi);
        slope_y = r * sinf(phi);
        return;
    }

    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1.0f / tanTheta;
    float G1 = 2.0f / (1.0f + sqrtf(1.0f + 1.0f / (a * a)));

    // Sample slope_x.
    float A = 2.0f * U1 / G1 - 1.0f;
    float tmp = 1.0f / (A * A - 1.0f);
    if (tmp > 1e10f) {
        tmp = 1e10f;
    }

    float B = tanTheta;
    float D = sqrtf(fmaxf(
        B * B * tmp * tmp - (A * A - B * B) * tmp,
        0.0f));

    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;

    slope_x = (A < 0.0f || slope_x_2 > 1.0f / tanTheta)
        ? slope_x_1
        : slope_x_2;

    // Sample slope_y.
    float S;
    if (U2 > 0.5f) {
        S = 1.0f;
        U2 = 2.0f * (U2 - 0.5f);
    }
    else {
        S = -1.0f;
        U2 = 2.0f * (0.5f - U2);
    }

    float z =
        (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.0f)
            + 0.597999f);

    slope_y = S * z * sqrtf(1.0f + slope_x * slope_x);
}


static __host__ __device__ glm::vec3 TrowbridgeReitzSample(
    const glm::vec3& wo,
    float alpha,
    float U1,
    float U2)
{
    // Stretch the direction..
    glm::vec3 woStretched = glm::normalize(
        glm::vec3(alpha * wo.x, alpha * wo.y, wo.z));

    // Sample slopes using reference-output version.
    float slope_x, slope_y;
    TrowbridgeReitzSample11(
        woStretched.z, U1, U2, slope_x, slope_y);

    // Compute azimuth of the stretched direction.
    float xyLength = sqrtf(
        woStretched.x * woStretched.x +
        woStretched.y * woStretched.y);

    float cosPhi = 1.0f;
    float sinPhi = 0.0f;

    if (xyLength > 0.0f) {
        cosPhi = woStretched.x / xyLength;
        sinPhi = woStretched.y / xyLength;
    }

    // Rotate the sampled slopes.
    float tmp = cosPhi * slope_x - sinPhi * slope_y;
    slope_y = sinPhi * slope_x + cosPhi * slope_y;
    slope_x = tmp;

    // Unstretch the slopes
    slope_x *= alpha;
    slope_y *= alpha;

    // Convert slopes into a microfacet normal.
    return glm::normalize(glm::vec3(-slope_x, -slope_y, 1.0f));
}


static __host__ __device__ glm::vec3 sampleGGXVisibleNormal(
    const glm::vec3& wo,
    float alpha,
    float U1,
    float U2)
{
    // Protect slope sampler from a rounded uniform value being == 1.
    U1 = fminf(fmaxf(U1, 0.0f), 0.99999994f);
    U2 = fminf(fmaxf(U2, 0.0f), 0.99999994f);
    bool flip = wo.z < 0.0f;

    glm::vec3 wh = TrowbridgeReitzSample(
        flip ? -wo : wo, alpha, U1, U2);

    return flip ? -wh : wh;
}



// Informed from PBRT microfacet.cpp:
// Fresnel, Oren-Nayar, GGX reflection, visible-normal PDF, and BSDF mix
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 1.0f / kPi;

__host__ __device__ bool hasPositive(const glm::vec3& v)
{
    return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f;
}

__host__ __device__ bool finiteFloat(float x)
{
    return x >= -FLT_MAX && x <= FLT_MAX; // Also rejects not number
}

__host__ __device__ bool finiteVector(const glm::vec3& v)
{
    return finiteFloat(v.x) && finiteFloat(v.y) && finiteFloat(v.z);
}

struct SurfaceFrame {
    glm::vec3 x, y, z;

    __host__ __device__ glm::vec3 toLocal(const glm::vec3& v) const
    {
        return glm::vec3(glm::dot(v, x), glm::dot(v, y), glm::dot(v, z));
    }
    __host__ __device__ glm::vec3 toWorld(const glm::vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }
};

__host__ __device__ SurfaceFrame makeFrame(const glm::vec3& normal)
{
    SurfaceFrame frame;
    frame.z = normal;
    const glm::vec3 helper = fabsf(normal.z) < 0.999f
        ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    frame.x = glm::normalize(glm::cross(helper, frame.z));
    frame.y = glm::cross(frame.z, frame.x);
    return frame;
}

__host__ __device__ void componentProbabilities(
    const Material& m, float& qd, float& qs)
{
    const bool diffuse = m.type == MaterialType::Standard && hasPositive(m.Kd);
    const bool glossy = hasPositive(m.Ks);
    qd = diffuse ? (glossy ? 0.5f : 1.0f) : 0.0f;
    qs = glossy ? (diffuse ? 0.5f : 1.0f) : 0.0f;
}

__host__ __device__ float fresnelDielectric(
    float cosThetaI, float etaI, float etaT)
{
    cosThetaI = fminf(fmaxf(cosThetaI, -1.0f), 1.0f);
    if (cosThetaI < 0.0f) {
        const float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = -cosThetaI;
    }
    if (etaI == etaT) return 0.0f;

    const float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    const float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0f) return 1.0f;
    const float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    const float rParallel = (etaT * cosThetaI - etaI * cosThetaT) /
                            (etaT * cosThetaI + etaI * cosThetaT);
    const float rPerpendicular = (etaI * cosThetaI - etaT * cosThetaT) /
                                 (etaI * cosThetaI + etaT * cosThetaT);
    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

__host__ __device__ glm::vec3 fresnelConductor(
    float cosThetaI, const glm::vec3& eta, const glm::vec3& k)
{
    // eta and k are relative to  air thats etaI = 1
    cosThetaI = fminf(fabsf(cosThetaI), 1.0f);
    const float cos2 = cosThetaI * cosThetaI;
    const float sin2 = 1.0f - cos2;
    const glm::vec3 eta2 = eta * eta;
    const glm::vec3 k2 = k * k;
    const glm::vec3 t0 = eta2 - k2 - glm::vec3(sin2);
    const glm::vec3 a2plusb2 = glm::sqrt(t0 * t0 + 4.0f * eta2 * k2);
    const glm::vec3 t1 = a2plusb2 + glm::vec3(cos2);
    const glm::vec3 a = glm::sqrt(glm::max(
        0.5f * (a2plusb2 + t0), glm::vec3(0.0f)));
    const glm::vec3 t2 = 2.0f * cosThetaI * a;
    const glm::vec3 rs = (t1 - t2) / (t1 + t2);
    const glm::vec3 t3 = cos2 * a2plusb2 + glm::vec3(sin2 * sin2);
    const glm::vec3 t4 = t2 * sin2;
    const glm::vec3 rp = rs * (t3 - t4) / (t3 + t4);
    return 0.5f * (rp + rs);
}

__host__ __device__ float ggxD(const glm::vec3& wh, float alpha)
{
    if (wh.z <= 0.0f) return 0.0f;
    const float alpha2 = alpha * alpha;
    // Isotropic PBRT D, rearranged to avoid tan(theta) and cos(theta)^4.
    const float denom = wh.x * wh.x + wh.y * wh.y + alpha2 * wh.z * wh.z;
    return alpha2 / (kPi * denom * denom);
}

__host__ __device__ float ggxLambda(const glm::vec3& w, float alpha)
{
    const float cosTheta = fabsf(w.z);
    if (cosTheta == 0.0f) return INFINITY;
    const float sin2 = fmaxf(0.0f, 1.0f - cosTheta * cosTheta);
    // Equivalent to (sqrt(1 + alpha^2 tan^2(theta)) - 1) / 2.
    return 0.5f * (sqrtf(cosTheta * cosTheta + alpha * alpha * sin2)
                   / cosTheta - 1.0f);
}

__host__ __device__ float ggxG1(const glm::vec3& w, float alpha)
{
    if (w.z <= 0.0f) return 0.0f;
    return 1.0f / (1.0f + ggxLambda(w, alpha));
}

__host__ __device__ float ggxG(
    const glm::vec3& wo, const glm::vec3& wi, float alpha)
{
    if (wo.z <= 0.0f || wi.z <= 0.0f) return 0.0f;
    // PBRt Smith masking-shadowing, not G1(wo) * G1(wi).
    return 1.0f / (1.0f + ggxLambda(wo, alpha) + ggxLambda(wi, alpha));
}

__host__ __device__ float ggxVisibleNormalPdf(
    const glm::vec3& wo, const glm::vec3& wh, float alpha)
{
    const float woDotWh = glm::dot(wo, wh);
    if (wo.z <= 0.0f || wh.z <= 0.0f || woDotWh <= 0.0f) return 0.0f;
    return ggxD(wh, alpha) * ggxG1(wo, alpha) * woDotWh / wo.z;
}

__host__ __device__ float glossyPdf(
    const glm::vec3& wo, const glm::vec3& wi, float alpha)
{
    if (wo.z <= 0.0f || wi.z <= 0.0f) return 0.0f;
    const glm::vec3 wh = glm::normalize(wo + wi);
    const float woDotWh = glm::dot(wo, wh);
    if (woDotWh <= 0.0f) return 0.0f;
    // Change of variables from visible microfacet normal to reflected ray.
    return ggxVisibleNormalPdf(wo, wh, alpha) / (4.0f * woDotWh);
}

__host__ __device__ glm::vec3 evaluateDiffuse(
    const Material& m, const glm::vec3& wo, const glm::vec3& wi)
{

    
    constexpr float invPi = 1.0f / 3.14159265358979323846f;

    if (m.diffuseSigmaDegrees == 0.0f)
        return m.Kd * invPi;

    const float sigma = m.diffuseSigmaDegrees * (kPi / 180.0f);
    const float sigma2 = sigma * sigma;
    const float A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
    const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
    const float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - wi.z * wi.z));
    const float sinThetaO = sqrtf(fmaxf(0.0f, 1.0f - wo.z * wo.z));
    float maxCos = 0.0f;
    if (sinThetaI > 1e-4f && sinThetaO > 1e-4f) {
        maxCos = fminf(1.0f, fmaxf(0.0f,
            (wi.x * wo.x + wi.y * wo.y) / (sinThetaI * sinThetaO)));
    }
    const float sinAlpha = wi.z > wo.z ? sinThetaO : sinThetaI;
    const float tanBeta = wi.z > wo.z ? sinThetaI / wi.z : sinThetaO / wo.z;
    return m.Kd * (kInvPi * (A + B * maxCos * sinAlpha * tanBeta));
}

__host__ __device__ glm::vec3 evaluateGlossy(
    const Material& m, const glm::vec3& wo, const glm::vec3& wi, bool frontFace)
{
    const glm::vec3 wh = glm::normalize(wo + wi);
    // Both local directions are above oriented reflection surface
    const float microfacetCos = glm::dot(wi, wh);
    const glm::vec3 F = m.type == MaterialType::Metal
        ? fresnelConductor(microfacetCos, m.metal.etaT, m.metal.k)
        : glm::vec3(fresnelDielectric(microfacetCos,
            frontFace ? 1.0f : m.ior, frontFace ? m.ior : 1.0f));
    return m.Ks * F * (ggxD(wh, m.alpha) * ggxG(wo, wi, m.alpha) /
                       (4.0f * wo.z * wi.z));
}

__host__ __device__ void absorb(PathSegment& path)
{
    path.color = glm::vec3(0.0f);
    path.remainingBounces = 0;
}

} // namespace

namespace bsdf {

__host__ __device__ Sample sampleSmoothDielectric(
    const Material& m, const glm::vec3& wo, bool frontFace, float u)
{
    Sample result{};
    if (!(wo.z > 0.0f) || !finiteVector(wo)) return result;

    const float etaIncident = frontFace ? 1.0f : m.transmission.ior;
    const float etaTransmitted = frontFace ? m.transmission.ior : 1.0f;
    const float eta = etaIncident / etaTransmitted;
    const float cosI = fminf(wo.z, 1.0f);
    const float sinT = eta * sqrtf(fmaxf(0.0f, 1.0f - cosI * cosI));
    const bool tir = sinT >= 1.0f;
    // dielectric Fresnel used by Standard glossy reflection.
    const float F = tir ? 1.0f : fresnelDielectric(cosI, etaIncident, etaTransmitted);
    result.delta = true;
    if (F >= 1.0f || u < F) {
        result.wi = glm::vec3(-wo.x, -wo.y, wo.z);
        result.branchProbability = F;
        result.deltaWeight = glm::vec3(1.0f); // F / P(reflect) = 1
    }
    else {
        const float cosT = sqrtf(fmaxf(0.0f, 1.0f - sinT * sinT));
        result.wi = glm::vec3(-eta * wo.x, -eta * wo.y, -cosT);
        result.transmission = true;
        result.branchProbability = 1.0f - F;
        // PBRT radiance transport: (1-F)/P(transmit) cancels, so just eta^2.
        result.deltaWeight = glm::vec3(eta * eta);
    }
    result.wi = glm::normalize(result.wi);
    return result;
}

__host__ __device__ glm::vec3 evaluate(
    const Material& m, const glm::vec3& wo, const glm::vec3& wi, bool frontFace)
{
    if (m.type == MaterialType::Transmission || wo.z <= 0.0f || wi.z <= 0.0f)
        return glm::vec3(0.0f);
    glm::vec3 f(0.0f);
    if (m.type == MaterialType::Standard && hasPositive(m.Kd)) {
        f += evaluateDiffuse(m, wo, wi);
    }
    if (hasPositive(m.Ks)) f += evaluateGlossy(m, wo, wi, frontFace);
    return f;
}

__host__ __device__ float pdf(
    const Material& m, const glm::vec3& wo, const glm::vec3& wi)
{
    if (m.type == MaterialType::Transmission || wo.z <= 0.0f || wi.z <= 0.0f)
        return 0.0f;
    float qd, qs;
    componentProbabilities(m, qd, qs);
    const float pd = wi.z * kInvPi;
    const float ps = qs > 0.0f ? glossyPdf(wo, wi, m.alpha) : 0.0f;
    return qd * pd + qs * ps;
}

__host__ __device__ Sample sample(
    const Material& m, const glm::vec3& wo, bool frontFace,
    thrust::default_random_engine& rng)
{
    Sample result{glm::vec3(0.0f), glm::vec3(0.0f), 0.0f};
    if (wo.z <= 0.0f) return result;
    if (m.type == MaterialType::Transmission) {
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        return sampleSmoothDielectric(m, wo, frontFace, u01(rng));
    }
    float qd, qs;
    componentProbabilities(m, qd, qs);
    if (qd == 0.0f && qs == 0.0f) return result;

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const bool chooseDiffuse = qd > 0.0f && (qs == 0.0f || u01(rng) < qd);
    if (chooseDiffuse) {
        result.wi = glm::normalize(
            sampleCosineWeightedHemisphere(glm::vec3(0.0f, 0.0f, 1.0f), rng));
    }
    else {
        // Separate statements make the RNG draw order not ambiguous n C++.
        const float u1 = u01(rng);
        const float u2 = u01(rng);
        const glm::vec3 wh = sampleGGXVisibleNormal(wo, m.alpha, u1, u2);
        const float woDotWh = glm::dot(wo, wh);
        if (!(woDotWh > 0.0f) || !finiteVector(wh)) return result;
        result.wi = glm::normalize(-wo + 2.0f * woDotWh * wh);
    }

    // rejected probability is part of  original sampler.
    // Dont not loop, resample, or renormalize the PDF over accepted directions.
    if (!(result.wi.z > 0.0f) || !finiteVector(result.wi)) return result;
    result.pdf = bsdf::pdf(m, wo, result.wi);
    result.f = bsdf::evaluate(m, wo, result.wi, frontFace);
    if (!(result.pdf > 0.0f) || !finiteFloat(result.pdf) || !finiteVector(result.f)) {
        result.pdf = 0.0f;
        result.f = glm::vec3(0.0f);
    }
    return result;
}

} // namespace bsdf

__host__ __device__ void scatterRay(
    PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal,
    const Material& m, thrust::default_random_engine& rng)
{
    // The shading kernel has already checked for a terminal emissive hit.
    // With one surface query left, scattering cannot reach another light.
    if (pathSegment.remainingBounces <= 1) {
        absorb(pathSegment);
        return;
    }

    const glm::vec3 geometricNormal = glm::normalize(normal);
    const glm::vec3 woWorld = -glm::normalize(pathSegment.ray.direction);
    const bool frontFace = glm::dot(geometricNormal, woWorld) >= 0.0f;
    const glm::vec3 shadingNormal = frontFace ? geometricNormal : -geometricNormal;
    const SurfaceFrame frame = makeFrame(shadingNormal);
    const glm::vec3 wo = glm::normalize(frame.toLocal(woWorld));
    const bsdf::Sample s = bsdf::sample(m, wo, frontFace, rng);
    if ((s.delta ? !(s.branchProbability > 0.0f) : !(s.pdf > 0.0f)) ||
        !finiteVector(s.wi)) {
        absorb(pathSegment);
        return;
    }

    pathSegment.color *= s.delta ? s.deltaWeight : s.f * (fabsf(s.wi.z) / s.pdf);
    if (!finiteVector(pathSegment.color) || !hasPositive(pathSegment.color)) {
        absorb(pathSegment);
        return;
    }

    const glm::vec3 wiWorld = glm::normalize(frame.toWorld(s.wi));
    const float side = glm::dot(wiWorld, shadingNormal);
    if (s.transmission ? !(side < 0.0f) : !(side > 0.0f)) {
        absorb(pathSegment);
        return;
    }
    pathSegment.ray.direction = wiWorld;
    // Offset only at ray spawning, using the geometric normal on the new side.
    const glm::vec3 offsetNormal = glm::dot(wiWorld, geometricNormal) > 0.0f
        ? geometricNormal : -geometricNormal;
    pathSegment.ray.origin = intersect + offsetNormal * EPSILON;
    --pathSegment.remainingBounces;
}
