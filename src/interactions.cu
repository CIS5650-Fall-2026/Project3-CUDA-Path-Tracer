#include "glm/detail/func_geometric.hpp"
#include "glm/detail/type_vec.hpp"
#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng,
    PathSegment &path)
{
#if HALTON_SAMPLING
    float u1 = sample1D(path);
    float u2 = sample1D(path);
    (void)rng;
#else
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u1 = u01(rng);
    float u2 = u01(rng);
    (void)path;
#endif

    float up = sqrt(u1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u2 * TWO_PI;

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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    bool outside)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
#if !REFRACTION
    (void)outside;
#endif

    // for diffuse material type
    if(m.hasReflective == 0.0f && m.hasRefractive == 0.0f) {
            // throughput
        pathSegment.color *= m.color; 

        // new ray, where the bounce starts
        pathSegment.ray.origin = intersect;  

        // new direction from the BSDF diffuse material
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng, pathSegment); // for diffuse material bounce
    }
#if REFRACTION
    // for refractive material type
    else if(m.hasReflective == 1.0f && m.hasRefractive == 1.0f) {
        glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        glm::vec3 n = glm::normalize(normal);

        // Air is 1. Entering glass uses 1/IOR. Leaving glass uses IOR/1.
        float etaIncident = outside ? 1.0f : m.indexOfRefraction;
        float etaTransmit = outside ? m.indexOfRefraction : 1.0f;
        float eta = etaIncident / etaTransmit;

        // Schlick. cosTheta is the angle between the ray and the normal facing it.
        float cosTheta = glm::clamp(glm::dot(-incident, n), 0.0f, 1.0f);
        float r0 = (etaIncident - etaTransmit) / (etaIncident + etaTransmit);
        r0 = r0 * r0;
        float oneMinusCos = 1.0f - cosTheta;
        float fresnel = r0 + (1.0f - r0) * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;

#if HALTON_SAMPLING
        float uReflect = sample1D(pathSegment);
        sample1D(pathSegment);
#else
        thrust::uniform_real_distribution<float> u01(0, 1);
        float uReflect = u01(rng);
#endif
        glm::vec3 refracted = glm::refract(incident, n, eta);
        // glm::refract returns 0 on total internal reflection.
        if (uReflect < fresnel || glm::dot(refracted, refracted) < 1e-6f) {
            pathSegment.ray.direction = glm::reflect(incident, n);
        } else {
            pathSegment.ray.direction = refracted;
        }

        pathSegment.color *= m.color;
        // Step off the surface along the new ray so the next trace does not hit this same point.
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
    }
#endif
    // Specular. With refraction disabled, glass takes this path and becomes a mirror.
    else if(m.hasReflective == 1.0f) {
#if HALTON_SAMPLING
        // Perfect specular consumes no random numbers. Advance two dimensions
        // so a later diffuse bounce stays on the same Halton coordinates.
        sample1D(pathSegment);
        sample1D(pathSegment);
        (void)rng;
#endif
        pathSegment.color *= m.color;
        pathSegment.ray.direction = glm::reflect(glm::normalize(pathSegment.ray.direction), glm::normalize(normal));
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
    }

    pathSegment.remainingBounces--; 

    // after hitting max bounce and haven't hit a light yet, set the color to black
    if (pathSegment.remainingBounces == 0) {
        pathSegment.color = glm::vec3(0.0f); 
    }


}
