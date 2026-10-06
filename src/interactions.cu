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

// Smith G1 for GGX, w in local frame (z = normal)
__host__ __device__ float smithG1(glm::vec3 w, float a) {
    float z2 = w.z * w.z;
    float t2 = fmaxf(0.0f, 1.0f - z2) / fmaxf(z2, 1e-8f);
    return 2.0f / (1.0f + sqrtf(1.0f + a * a * t2));
}

// Sample a GGX visible normal (Heitz 2018), Ve in local frame
__host__ __device__ glm::vec3 sampleGGXVNDF(glm::vec3 Ve, float a, float u1, float u2) {
    glm::vec3 Vh = glm::normalize(glm::vec3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    glm::vec3 T1 = lensq > 0.0f ? glm::vec3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : glm::vec3(1, 0, 0);
    glm::vec3 T2 = glm::cross(Vh, T1);
    float r = sqrtf(u1), phi = TWO_PI * u2;
    float t1 = r * cosf(phi), t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
    glm::vec3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    return glm::normalize(glm::vec3(a * Nh.x, a * Nh.y, fmaxf(0.0f, Nh.z)));
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 wi = glm::normalize(pathSegment.ray.direction);
    glm::vec3 n = glm::dot(wi, normal) > 0.0f ? -normal : normal;

    if (m.hasRefractive > 0.0f) {
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Rough glass: refract/reflect about a sampled GGX microfacet normal
        glm::vec3 mn = n;
        if (m.roughness > 0.0f) {
            float a = fmaxf(m.roughness * m.roughness, 1e-3f);
            glm::vec3 t = glm::normalize(glm::cross(n, fabsf(n.x) > 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)));
            glm::vec3 b = glm::cross(n, t);
            glm::vec3 woL = glm::normalize(glm::vec3(glm::dot(-wi, t), glm::dot(-wi, b), fmaxf(glm::dot(-wi, n), 1e-4f)));
            glm::vec3 hL = sampleGGXVNDF(woL, a, u01(rng), u01(rng));
            mn = hL.x * t + hL.y * b + hL.z * n;
        }

        float ior = m.indexOfRefraction;
        float eta = outside ? (1.0f / ior) : ior;
        // Detect TIR ourselves: GLM 0.9.6 refract() returns NaN (sqrt(k<0) * 0) instead of zero
        float cosI = -glm::dot(wi, mn);
        float k = 1.0f - eta * eta * (1.0f - cosI * cosI);
        bool totalInternalReflection = k < 0.0f;
        glm::vec3 refracted = totalInternalReflection ? glm::vec3(0.0f) : glm::refract(wi, mn, eta);

        float cosTheta = outside ? -glm::dot(wi, mn) : -glm::dot(refracted, mn);
        float r0 = (1.0f - ior) / (1.0f + ior);
        r0 = r0 * r0;
        float fresnel = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);

        if (totalInternalReflection || u01(rng) < fresnel) {
            pathSegment.ray.direction = glm::reflect(wi, mn);
            pathSegment.ray.origin = intersect + n * 0.001f;
        }
        else {
            pathSegment.ray.direction = refracted;
            pathSegment.ray.origin = intersect - n * 0.001f;
        }

        // Microfacet sample can send the ray to the wrong side: drop it
        bool reflected = glm::dot(pathSegment.ray.direction, n) > 0.0f;
        bool wentOut = glm::dot(pathSegment.ray.origin - intersect, n) > 0.0f;
        if (reflected != wentOut) pathSegment.color = glm::vec3(0.0f);

        pathSegment.color *= m.specular.color;
        pathSegment.lastPdf = -1.0f;
    }
    else if (m.hasReflective > 0.0f && m.roughness <= 0.0f) {
        pathSegment.ray.direction = glm::reflect(wi, n);
        pathSegment.ray.origin = intersect + n * 0.001f;
        pathSegment.color *= m.specular.color;
        pathSegment.lastPdf = -1.0f;
    }
    else if (m.hasReflective > 0.0f) {
        // GGX metal, or glossy coat over a diffuse base
        thrust::uniform_real_distribution<float> u01(0, 1);
        float a = fmaxf(m.roughness * m.roughness, 1e-3f);
        glm::vec3 t = glm::normalize(glm::cross(n, fabsf(n.x) > 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)));
        glm::vec3 b = glm::cross(n, t);
        glm::vec3 woL(glm::dot(-wi, t), glm::dot(-wi, b), fmaxf(glm::dot(-wi, n), 1e-4f));
        woL = glm::normalize(woL);

        float pSpec = m.metallic > 0.5f ? 1.0f : 0.35f;
        if (u01(rng) < pSpec) {
            glm::vec3 hL = sampleGGXVNDF(woL, a, u01(rng), u01(rng));
            glm::vec3 wiL = glm::reflect(-woL, hL);
            if (wiL.z <= 0.0f) {
                pathSegment.color = glm::vec3(0.0f);   // sampled below the surface
                pathSegment.ray.direction = n;
            }
            else {
                glm::vec3 F0 = m.metallic > 0.5f ? m.color : glm::vec3(0.04f);
                glm::vec3 F = F0 + (1.0f - F0) * powf(1.0f - glm::dot(wiL, hL), 5.0f);
                // VNDF sampling: f * cos / pdf = F * G1(wi)
                pathSegment.color *= F * smithG1(wiL, a) / pSpec;
                pathSegment.ray.direction = wiL.x * t + wiL.y * b + wiL.z * n;
            }
        }
        else {
            // Diffuse base, attenuated by the coat's Fresnel
            float Fo = 0.04f + 0.96f * powf(1.0f - woL.z, 5.0f);
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(n, rng);
            pathSegment.color *= m.color * (1.0f - Fo) / (1.0f - pSpec);
        }
        pathSegment.ray.origin = intersect + n * 0.001f;
        pathSegment.lastPdf = -1.0f;   // glossy lobes skip MIS for now
    }
    else {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(n, rng);
        pathSegment.lastPdf = glm::dot(pathSegment.ray.direction, n) / PI;
        pathSegment.ray.origin = intersect + n * 0.001f;
        pathSegment.color *= m.color;
    }
}
