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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool frontFace,
    const Material &m,
    thrust::default_random_engine &rng)
{
  // TODO: implement this.
  // A basic implementation of pure-diffuse shading will just call the
  // calculateRandomDirectionInHemisphere defined above.
  
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

  if (m.hasRefractive > 0.0f) {
    glm::vec3 incoming = pathSegment.ray.direction;

    // eta in pbr is IOR -> this is incoming
    float n1 = frontFace ? 1.0f : m.indexOfRefraction;
    float n2 = frontFace ? m.indexOfRefraction : 1.0;
    
    float R0 = powf((n1 - n2) / (n1 + n2), 2.0f);

    float cosTheta = glm::clamp(glm::dot(-incoming, normal), 0.0f, 1.0f);

    float R = R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0f);

    float etaRatio = n1 / n2;
    float sin2ThetaTransmitted = etaRatio * etaRatio * (1.0f - cosTheta * cosTheta);

    // u01 < R from 9.5
    if (sin2ThetaTransmitted >= 1.0f || u01(rng) < R) {
      pathSegment.ray.direction = glm::normalize(glm::reflect(incoming, normal));
    }
    else {
      pathSegment.ray.direction = glm::normalize(glm::refract(incoming, normal, etaRatio));
      // Color correction from 9.5
      pathSegment.color *= etaRatio * etaRatio;
    }
  } else if (m.hasReflective > 0.0f) {
    // glm::reflect wr (omega_r) for us
    pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
  }
  else {
    // Valid / works for a scene with only diffuse materials
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
  }

  // A 'bounced' ray starts at the intersection point. Offset slightly above the surface to avoid it being in the surface.
  // Added offset sign for refraction support
  float offsetSign = glm::dot(pathSegment.ray.direction, normal) >= 0.0f ? 1.0f : -1.0f;
  pathSegment.ray.origin = intersect + offsetSign * (normal * 0.0001f);

  // Accumulate color from all rays in path
  pathSegment.color *= m.color;

  pathSegment.remainingBounces--;

}
