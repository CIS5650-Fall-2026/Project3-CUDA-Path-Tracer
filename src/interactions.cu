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

__host__ __device__
glm::vec3 sampleReflection(
  glm::vec3 rayDir,
  glm::vec3 normal,
  float roughness,
  thrust::default_random_engine& rng)
{
  if (roughness <= 0.0f) {
    return glm::normalize(glm::reflect(rayDir, normal));
  }

  thrust::uniform_real_distribution<float> u01(0, 1);

  float clampedRoughness = max(0.001f, roughness);
  float exponent = (2.0f / (clampedRoughness * clampedRoughness)) - 2.0f;

  float xi1 = u01(rng);
  float xi2 = u01(rng);

  float cosTheta = pow(xi1, 1.0f / (exponent + 1.0f));
  float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
  float phi = 2 * PI * xi2;

  float xs = cos(phi) * sinTheta;
  float ys = sin(phi) * sinTheta;
  float zs = cosTheta;

  glm::vec3 perfectReflection = glm::normalize(glm::reflect(rayDir, normal));

  glm::vec3 up = glm::vec3(1.0f, 0.0f, 0.0f);
  if (abs(perfectReflection.z) < 0.999f) {
    up = glm::vec3(0.0f, 0.0f, 1.0f);
  }

  glm::vec3 tangent = glm::normalize(glm::cross(up, perfectReflection));
  glm::vec3 bitangent = glm::cross(perfectReflection, tangent);

  return glm::normalize(tangent * xs + bitangent * ys + perfectReflection * zs);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    if (glm::length(m.emittance) > 0.0f) {
      // if light, end
      pathSegment.remainingBounces = 0;
      pathSegment.color *= m.emittance;
      return;
    }
    pathSegment.remainingBounces -= 1;

    glm::vec3 rayDir = glm::normalize(pathSegment.ray.direction);
    bool entering = glm::dot(rayDir, normal) < 0.0f;
    glm::vec3 N = entering ? normal : -normal;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float r = u01(rng);

    if (r < m.metallic) {
      // metallic reflection
      pathSegment.color *= m.color;
      pathSegment.ray.direction = sampleReflection(rayDir, N, m.roughness, rng);
      pathSegment.ray.origin = intersect + N * 0.0001f;
    } 
    else 
    {
      // calculate fresnel
      float etaI = entering ? 1.0f : m.indexOfRefraction;
      float etaT = entering ? m.indexOfRefraction : 1.0f;
      float eta = etaI / etaT;

      float cosTheta = glm::clamp(glm::dot(-rayDir, N), 0.0f, 1.0f);
      float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
      float F;

      if (eta * sinTheta > 1.0f) {
        F = 1.0f;
      }
      else {
        float r0 = (etaI - etaT) / (etaI + etaT);
        r0 *= r0;
        F = r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5.0f);
      }

      float pReflect = F;
      float pTransmit = (1.0f - F) * m.transmission;
      float r = u01(rng);

      if (r < pReflect) {
        // dielectric reflection
        pathSegment.ray.direction = sampleReflection(rayDir, N, m.roughness, rng);
        pathSegment.ray.origin = intersect + N * 0.001f;
      }
      else if (r < pReflect + pTransmit) {
        // refraction
        pathSegment.color *= m.color;
        pathSegment.ray.direction = glm::normalize(glm::refract(rayDir, N, eta));
        pathSegment.ray.origin = intersect - N * 0.001f;
      }
      else {
        // diffuse
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(N, rng);
        pathSegment.ray.origin = intersect + N * 0.001f;
      }
    }
}
