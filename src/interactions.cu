#include "interactions.h"

#include "dielectric.h"
#include "intersections.h"
#include "microfacet.h"
#include "nee.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3
calculateRandomDirectionInHemisphere(glm::vec3 normal,
                                     thrust::default_random_engine &rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = sqrt(u01(rng));      // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  } else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
      glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
      glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal + cos(around) * over * perpendicularDirection1 +
         sin(around) * over * perpendicularDirection2;
}

__device__ glm::vec3 evaluate_bsdf_for_direction(
    const Material& m,
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    bool outside) {
    bool smooth = m.roughness_factor <= 0.001f;
    float wDiffuse;
    float wMetal;
    float wTransmission;
    Microfacet::material_lobe_weights(m, wDiffuse, wMetal, wTransmission);

    glm::vec3 value(0.0f);
    if (wDiffuse > 0.0f && glm::dot(normal, woWorld) > 0.0f &&
        glm::dot(normal, wiWorld) > 0.0f) {
        value += wDiffuse * m.color / PI;
    }

    if (!smooth && wMetal > 0.0f) {
        value += wMetal *
                 Microfacet::evaluate_brdf(woWorld, wiWorld, normal, m,
                                            Microfacet::material_alpha(m));
    }

    if (!smooth && wTransmission > 0.0f) {
        value += wTransmission *
                 Microfacet::evaluate_dielectric_bsdf(
                     woWorld, wiWorld, normal, m, outside,
                     Microfacet::material_alpha(m));
    }

    return value;
}

__device__ float evaluate_bsdf_pdf_for_direction(
    const Material& m,
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    bool outside) {
    bool smooth = m.roughness_factor <= 0.001f;
    float wDiffuse;
    float wMetal;
    float wTransmission;
    Microfacet::material_lobe_weights(m, wDiffuse, wMetal, wTransmission);

    float pdf = 0.0f;
    if (wDiffuse > 0.0f && glm::dot(normal, woWorld) > 0.0f &&
        glm::dot(normal, wiWorld) > 0.0f) {
        pdf += wDiffuse * fmaxf(glm::dot(normal, wiWorld), 0.0f) / PI;
    }

    if (!smooth && wMetal > 0.0f) {
        pdf += wMetal * Microfacet::evaluate_pdf(
                            woWorld, wiWorld, normal,
                            Microfacet::material_alpha(m));
    }

    if (!smooth && wTransmission > 0.0f) {
        pdf += wTransmission *
                   Microfacet::evaluate_dielectric_pdf(
                   woWorld, wiWorld, normal, m, outside,
                   Microfacet::material_alpha(m));
    }

    return pdf;
}

__device__ ScatterResult empty_scatter_result() {
    ScatterResult result;
    result.throughputMultiplier = glm::vec3(0.0f);
    result.contribution = glm::vec3(0.0f);
    result.etaScale = 1.0f;
    result.pdf = 0.0f;
    result.wasSpecular = false;
    result.wasTransmission = false;
    return result;
}

__device__ ScatterResult calculate_diffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    ScatterResult result = empty_scatter_result();
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect;
    float cosTheta = fmaxf(glm::dot(normal, pathSegment.ray.direction), 0.0f);
    result.throughputMultiplier = m.color;
    result.pdf = cosTheta / PI;
    return result;
}

__device__ ScatterResult scatterRay(PathSegment &pathSegment,
                                    glm::vec3 intersect, glm::vec3 normal,
                                    bool outside,
                                    const Material &m,
                                    SceneLight *lights,
                                    int lights_size,
                                    Geom *geoms,
                                    int geoms_size,
                                    Triangle *triangles,
                                    int triangles_size,
                                    BVHNode *bvh,
                                    Material *materials,
                                    int materials_size,
                                    int materialId,
                                    float totalLightArea,
                                    thrust::default_random_engine &rng) {
    const float materialEpsilon = 0.001f;
    (void)totalLightArea;
    
    ScatterResult result = empty_scatter_result();

    glm::vec3 woWorld = glm::normalize(-pathSegment.ray.direction);
    
    bool smooth =
        m.roughness_factor <= materialEpsilon;
    float alpha = Microfacet::material_alpha(m);

    bool pureMetal =
        m.metalic_factor >= 1.0f - materialEpsilon;

    bool pureGlass =
        m.metalic_factor <= materialEpsilon &&
        m.transmission_factor >= 1.0f - materialEpsilon;

    bool pureDeltaMetal =
        pureMetal && smooth;

    bool pureDeltaGlass =
        pureGlass && smooth;

    bool doMIS =
        !(pureDeltaMetal || pureDeltaGlass);

    glm::vec3 sampledWm = normal;
    if (!smooth && (m.metalic_factor > materialEpsilon ||
                    m.transmission_factor > materialEpsilon)) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec2 u(u01(rng), u01(rng));
        sampledWm = Microfacet::sample_wm(alpha, normal, u);
    }

    // determine if do NEE + MIS if pure delta
    if (doMIS) {
        NeeSample neeSample =
        Nee::get_nee(intersect, normal, lights, lights_size, triangles,
                         triangles_size, materials, materials_size,
                         totalLightArea, rng);
        if (neeSample.valid &&
            Nee::shadow_ray_visible(intersect, neeSample, geoms, geoms_size,
                                    triangles, triangles_size, bvh, materials,
                                    materials_size)) {
            glm::vec3 bsdf =
                evaluate_bsdf_for_direction(m, woWorld, neeSample.wi, normal,
                                            outside);
            float bsdfPdf =
                evaluate_bsdf_pdf_for_direction(m, woWorld, neeSample.wi,
                                                normal, outside);
            float cosSurface = fmaxf(glm::dot(normal, neeSample.wi), 0.0f);
            if (cosSurface > 0.0f && neeSample.pdfLight > 0.0f) {
                float misWeight =
                    Nee::power_heuristic(neeSample.pdfLight, bsdfPdf);
                result.contribution =
                    misWeight * neeSample.Li * bsdf * cosSurface /
                    neeSample.pdfLight;
            }
        }
    }

    // Case pure metal
    if (pureMetal) {
        ScatterResult sampledResult =
            Microfacet::calculate_reflection(pathSegment, intersect, normal, m,
                                             pureDeltaMetal, sampledWm, rng);
        sampledResult.contribution = result.contribution;
        return sampledResult;
    }

    // Case pure dielectric
    if (pureGlass) {
        ScatterResult sampledResult =
            Dielectric::calculate_transmission(pathSegment, intersect, normal,
                                               outside, m, smooth, woWorld,
                                               sampledWm, rng);
        sampledResult.contribution = result.contribution;

        if (sampledResult.wasTransmission) {
            if (outside && m.thicknessFactor > 0.0f) {
                pathSegment.mediumMaterialIndex = materialId;
            } else if (!outside) {
                pathSegment.mediumMaterialIndex = -1;
            }
        }

        return sampledResult;
    }

    if (m.transmission_factor < materialEpsilon &&
        m.metalic_factor < materialEpsilon) {
        ScatterResult sampledResult =
            calculate_diffuse(pathSegment, intersect, normal, m, rng);
        sampledResult.contribution = result.contribution;
        return sampledResult;
    }

    // Opaque metalic + dielectric mix
    float p_diffuse;
    float p_metal;
    float p_transmission;
    Microfacet::material_lobe_weights(m, p_diffuse, p_metal, p_transmission);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_chance = u01(rng);

    ScatterResult sampledResult = empty_scatter_result();
    if (random_chance < p_metal) {
        sampledResult =
            Microfacet::calculate_reflection(pathSegment, intersect, normal, m,
                                             pureDeltaMetal, sampledWm, rng);
    } else if (random_chance < p_metal + p_transmission) {
        sampledResult =
            Dielectric::calculate_transmission(pathSegment, intersect, normal,
                                               outside, m, smooth, woWorld,
                                               sampledWm, rng);
    } else {
        sampledResult = calculate_diffuse(pathSegment, intersect, normal, m, rng);
    }

    if (!sampledResult.wasSpecular) {
        glm::vec3 wiWorld = glm::normalize(pathSegment.ray.direction);
        float pdf =
            evaluate_bsdf_pdf_for_direction(m, woWorld, wiWorld, normal,
                                            outside);
        glm::vec3 bsdf =
            evaluate_bsdf_for_direction(m, woWorld, wiWorld, normal, outside);

        if (pdf <= 0.0f ||
            (bsdf.x == 0.0f && bsdf.y == 0.0f && bsdf.z == 0.0f)) {
            sampledResult = empty_scatter_result();
            sampledResult.contribution = result.contribution;
            return sampledResult;
        }

        sampledResult.pdf = pdf;
        sampledResult.throughputMultiplier =
            bsdf * fabsf(glm::dot(normal, wiWorld)) / pdf;
    }

    if (sampledResult.wasTransmission) {
        if (outside && m.thicknessFactor > 0.0f) {
            pathSegment.mediumMaterialIndex = materialId;
        } else if (!outside) {
            pathSegment.mediumMaterialIndex = -1;
        }
    }

    sampledResult.contribution = result.contribution;
    return sampledResult;
}
