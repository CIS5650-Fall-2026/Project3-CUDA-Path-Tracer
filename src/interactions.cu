#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#include <glm/gtx/norm.hpp>

// world space distance a scattered ray's origin is pushed off the surface
#define SCATTER_EPSILON 1e-3f

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

// GGX (Trowbridge-Reitz) microfacet reflection
// alpha = roughness^2 ("perceptual" roughness, as in Disney/UE4)
// we floor it to prevent divide by zero errors
__host__ __device__ static float ggxAlpha(float roughness)
{
    return glm::max(roughness * roughness, 1e-3f);
}

// Trowbridge-Reitz Normal Distribution Function
// returns relative density of microfacets pointing in a direction
__host__ __device__ static float ggxD(float cos2, float sin2, float alpha)
{
    float a2 = alpha * alpha;
    float d = a2 * cos2 + sin2;
    return a2 / (PI * d * d);
}

// Smith Lambda
// returns ratio of self-occluded microfacet area to visible microfacet
// along direction x
__host__ __device__ static float ggxLambda(float NdotX, float alpha)
{
    // cos(theta)^2, since dot prod is cosine
    float c2 = glm::max(NdotX * NdotX, 1e-8f);
    // 0.5 * (-1.0 + sqrt(1.0 + alpha^2 * tan(theta)^2))
    return 0.5f * (-1.0f + sqrtf(1.0f + alpha * alpha * (1.0f - c2) / c2));
}

// fresnel effect, making glancing angles have higher values
// we return a vector because the base reflectance (f0) of some materials depends
// on certain wavelengths (colors)
__host__ __device__ static glm::vec3 schlickFresnel(glm::vec3 f0, float cosTheta)
{
    return f0 + (1.0f - f0) * powf(1.0f - glm::clamp(cosTheta, 0.0f, 1.0f), 5.0f);
}

// Getting an orthonormal basis for n
// (local coordinate frame for n)
__host__ __device__ static void tangentFrame(glm::vec3 n, glm::vec3& t, glm::vec3& b)
{
    glm::vec3 notN = fabsf(n.x) < SQRT_OF_ONE_THIRD ? glm::vec3(1, 0, 0)
        : (fabsf(n.y) < SQRT_OF_ONE_THIRD ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
    t = glm::normalize(glm::cross(n, notN));
    b = glm::cross(n, t);
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals"
// returns a randomly sampled microfacet normal vector (h) in local coordinates
__host__ __device__ static glm::vec3 sampleGGXVNDF(glm::vec3 wo, float alpha, float u1, float u2)
{
    glm::vec3 Vh = glm::normalize(glm::vec3(alpha * wo.x, alpha * wo.y, wo.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    glm::vec3 T1 = lensq > 0.0f ? glm::vec3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 T2 = glm::cross(Vh, T1);
    // uniform disk sample, warped toward the visible half of the disk
    float r = sqrtf(u1);
    float phi = TWO_PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    // project onto the hemisphere, then unstretched
    glm::vec3 Nh = t1 * T1 + t2 * T2 + sqrtf(glm::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    return glm::normalize(glm::vec3(alpha * Nh.x, alpha * Nh.y, glm::max(0.0f, Nh.z)));
}

__host__ __device__ glm::vec3 evalBSDF(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi,
    float& pdf)
{
    float NdotL = glm::dot(normal, wi);
    if (!isMicrofacet(m))
    {
        // Lambertian, sampled by calculateRandomDirectionInHemisphere (cosine-weighted)
        pdf = glm::max(NdotL, 0.0f) / PI;
        return m.color / PI;
    }

    // if view ray or light ray behind normal, don't use it
    float NdotV = glm::dot(normal, wo);
    if (NdotL <= 0.0f || NdotV <= 0.0f)
    {
        pdf = 0.0f;
        return glm::vec3(0.0f);
    }

    float alpha = ggxAlpha(m.roughness);
    // half vector
    glm::vec3 h = glm::normalize(wo + wi);
    float NdotH = glm::max(glm::dot(normal, h), 0.0f);
    glm::vec3 hTangential = glm::cross(normal, h);
    float D = ggxD(NdotH * NdotH, glm::dot(hTangential, hTangential), alpha);
    float lambdaV = ggxLambda(NdotV, alpha);
    float lambdaL = ggxLambda(NdotL, alpha);

    // VNDF pdf of h, times the reflection Jacobian 1 / (4 wo.h): G1(wo) * D / (4 NdotV)
    pdf = D / (4.0f * NdotV * (1.0f + lambdaV));
    // Cook-Torrance: F * D * G2 / (4 NdotV NdotL)
    return schlickFresnel(m.color, glm::dot(wo, h)) * D / (4.0f * NdotV * NdotL * (1.0f + lambdaV + lambdaL));
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 geomNormal,
    const Material &m,
    thrust::default_random_engine &rng,
    bool outside)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.hasRefractive > 0.0f)
    {
        glm::vec3 I = glm::normalize(pathSegment.ray.direction);
        float eta = outside ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
        float cosThetaI = glm::dot(-I, normal);

        // Since glm::refract can produce NaN with tir,
        // we compute k using Snell's law, and check if it is valid
        // before refracting.
        float dotNI = glm::dot(normal, I);
        float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
        bool tir = k < 0.0f;

        // Schlick approximation
        float r0 = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
        r0 *= r0;

        // Since we want the cosine on the less dense side (air), we
        // check here if we are entering or exiting and use the proper cosine value.
        float cosSchlick = outside ? cosThetaI : glm::sqrt(glm::max(k, 0.0f));
        // we take into account a negative k value if tir is true.
        float fresnel = tir ? 1.0f : r0 + (1.0f - r0) * powf(1.0f - cosSchlick, 5.0f);

        if (tir || u01(rng) < fresnel)
        {
            pathSegment.ray.direction = glm::reflect(I, normal);
        }
        else
        {
            // No tir, so we won't get NaN
            // finish up the rest of Snell's law to get the refracted direction
            glm::vec3 refracted = eta * I - (eta * dotNI + glm::sqrt(k)) * normal;
            pathSegment.ray.direction = glm::normalize(refracted);
        }
        pathSegment.color *= m.color;
    }
    else if (isMicrofacet(m))
    {
        glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);
        glm::vec3 t, b;
        tangentFrame(normal, t, b);
        // wo, transformed to local "normal coords"
        glm::vec3 woLocal(glm::dot(wo, t), glm::dot(wo, b), glm::dot(wo, normal));
        float alpha = ggxAlpha(m.roughness);

        float u1 = u01(rng);
        float u2 = u01(rng);
        glm::vec3 hLocal = sampleGGXVNDF(woLocal, alpha, u1, u2);
        glm::vec3 h = hLocal.x * t + hLocal.y * b + hLocal.z * normal;
        glm::vec3 wi = glm::reflect(-wo, h);
        float NdotL = glm::dot(normal, wi);
        if (woLocal.z <= 0.0f || NdotL <= 0.0f)
        {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        float lambdaV = ggxLambda(woLocal.z, alpha);
        float lambdaL = ggxLambda(NdotL, alpha);
        pathSegment.ray.direction = wi;
        pathSegment.color *= schlickFresnel(m.color, glm::dot(wo, h)) * ((1.0f + lambdaV) / (1.0f + lambdaV + lambdaL));
        // solid-angle pdf of wi for MIS if this ray hits a light
        pathSegment.bsdfPdf = ggxD(hLocal.z * hLocal.z, hLocal.x * hLocal.x + hLocal.y * hLocal.y, alpha)
            / (4.0f * woLocal.z * (1.0f + lambdaV));
    }
    else if (m.hasReflective > 0.0f)
    {
        pathSegment.ray.direction = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
        pathSegment.color *= m.color;
    }
    else
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
        pathSegment.bsdfPdf = glm::max(glm::dot(normal, pathSegment.ray.direction), 0.0f) / PI;
    }

    // When we use bump map normals, there is a chance scattered rays can go inside the mesh.
    // To prevent this from messing up the lighting, we do a check with the actual geometry normals
    // to stop any of those stray rays from contributing light.
    float geomSide = glm::dot(pathSegment.ray.direction, geomNormal);
    if (m.hasRefractive <= 0.0f && geomSide <= 0.0f)
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    // We offset by a positive amount when bouncing off, and a negative amount when refracting.
    // This ensures the ray doesn't continuously bounce off a surface when it reaches it.
    pathSegment.ray.origin = intersect + geomNormal * (geomSide > 0.0f ? SCATTER_EPSILON : -SCATTER_EPSILON);
    --pathSegment.remainingBounces;
}
