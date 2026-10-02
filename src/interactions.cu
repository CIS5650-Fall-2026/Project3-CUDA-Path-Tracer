#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include "sampling.h"
#include "geometry.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng,
    float hemisphereSample, float azimuthSample)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(hemisphereSample >= 0 ? hemisphereSample : u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = (azimuthSample >= 0 ? azimuthSample : u01(rng)) * TWO_PI;

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
    thrust::default_random_engine &rng)
{
    // Scatter diffusely in a cosine-weighted hemisphere and apply surface color.
    pathSegment.ray.origin = intersect + rayEpsilon * normal;
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.color *= m.color;
}

// Explicit samples keep Halton dimensions stable
// after material sorting, stream compaction and checkpoint resume.
__host__ __device__ bool scatterRay(PathSegment& path, glm::vec3 point,
    glm::vec3 normal, bool frontFace, const Material& m,
    bool refraction, float u, float v)
{
    bool delta=m.hasReflective>0||(m.hasRefractive>0&&refraction);
    if(m.hasRefractive>0&&refraction) {
        float eta=frontFace?1.0f/m.indexOfRefraction:m.indexOfRefraction;
        float cosine=glm::clamp(-glm::dot(path.ray.direction,normal),0.0f,1.0f);
        float sin2=eta*eta*(1-cosine*cosine);
        float r0=(1-m.indexOfRefraction)/(1+m.indexOfRefraction);r0*=r0;
        float oneMinus=1-cosine;
        float fresnel=r0+(1-r0)*oneMinus*oneMinus*oneMinus*oneMinus*oneMinus;
        if(sin2>=1||u<fresnel) path.ray.direction=glm::reflect(path.ray.direction,normal);
        else {
            path.ray.direction=glm::refract(path.ray.direction,normal,eta);
            path.color*=eta*eta;path.etaScale/=eta*eta;
        }
    } else if(m.hasReflective>0) path.ray.direction=glm::reflect(path.ray.direction,normal);
    else {
        thrust::default_random_engine rng;
        path.ray.direction=calculateRandomDirectionInHemisphere(normal,rng,u,v);
    }
    path.ray.direction=glm::normalize(path.ray.direction);
    path.ray.origin=point+normal*(glm::dot(path.ray.direction,normal)>=0?rayEpsilon:-rayEpsilon);
    path.color*=m.color;
    path.previousPoint=point;path.previousDelta=delta;
    path.previousPdf=delta?0.0f:fmaxf(0,glm::dot(path.ray.direction,normal))/PI;
    return delta;
}

// Forward surface scattering to the explicit-sample ray-scattering overload.
__host__ __device__ bool scatterSurface(PathSegment& path, glm::vec3 point,
    glm::vec3 normal, bool frontFace, const Material& m,
    bool refraction, float u, float v)
{
    return scatterRay(path, point, normal, frontFace, m, refraction, u, v);
}
