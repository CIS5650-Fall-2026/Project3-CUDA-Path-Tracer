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

__host__ __device__ float schlickFresnel(
    float cosTheta,
    float etaI,
    float etaT
) {
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 *= r0;
    float c = 1.f - cosTheta;
    return r0 + (1.f - r0) * c * c * c * c * c;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    
    // emissive materials don't scatter
    if (m.emittance > 0.f) {
        pathSegment.color *= (m.emittance * m.color);
		pathSegment.remainingBounces = 0;
        return;
    }

    const float OFFSET = 1e-3f;
    glm::vec3 wi = pathSegment.ray.direction;
    glm::vec3 newDir;
    glm::vec3 newOrigin;

    if (m.hasRefractive > 0.f) {
        float etaI = outside ? 1.f : m.indexOfRefraction;
        float etaT = outside ? m.indexOfRefraction : 1.f;

        // find total internal reflection
        glm::vec3 refracted = glm::refract(wi, normal, etaI / etaT);
        bool interalTotal = glm::dot(refracted, refracted) < 1e-12f;
        float F = 1.f;

        if (!interalTotal) {
            // angle on less dense side
            float cosTheta = (etaI > etaT)
                ? glm::dot(refracted, -normal)
                : glm::dot(-wi, normal);

            F = (m.hasReflective > 0.f) ? schlickFresnel(cosTheta, etaI, etaT) : 0.f;
        }

        thrust::uniform_real_distribution<float> u01(0, 1);

        if (u01(rng) < F) {
            newDir = glm::reflect(wi, normal);
            newOrigin = intersect + normal * OFFSET;
        }
        else {
            newDir = refracted;
            newOrigin = intersect - normal * OFFSET;
        }
    }
    else if (m.hasReflective > 0.f) {
        newDir = glm::reflect(wi, normal);
        newOrigin = intersect + normal * OFFSET;
    }
    else {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        newOrigin = intersect + normal * OFFSET;
    }

    // add small epsilon to prevent self intersection
    pathSegment.ray.origin = newOrigin;
	pathSegment.ray.direction = glm::normalize(newDir);
	pathSegment.color *= m.color;
    pathSegment.remainingBounces = glm::max(0, pathSegment.remainingBounces - 1);
}

__host__ __device__ glm::vec4 sampleTexture(
    const Texture& tex,
    const glm::vec4* texels,
    glm::vec2 uv
) {
    // wrap uvs 0 -> 1
    uv = uv - glm::floor(uv);
    int x = glm::min((int)(uv.x * tex.width), tex.width - 1);
    int y = glm::min((int)((1.f - uv.y) * tex.height), tex.height - 1);

    return texels[tex.offset + y * tex.width + x];
}

__host__ __device__ glm::vec3 sampleEnvironment(
    const Texture& env,
    const glm::vec4* texels,
    glm::vec3 dir
) {
    float u = atan2f(dir.z, dir.x) / TWO_PI + 0.5f;
    float v = 0.5f + asinf(glm::clamp(dir.y, -1.f, 1.f)) / PI;

    return glm::vec3(sampleTexture(env, texels, glm::vec2(u, v)));
}