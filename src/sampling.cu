#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#include "interactions.h"
#include "intersections.h"
#include "Volume/volume.h"


__host__ __device__ void sampleRandomLight(Geom* geo, int num_geoms, VolumeData& vd, const Light* lights, int num_lights, const Material* materials, glm::vec3 intersect, glm::vec3 normal, glm::vec3 wo, const Material& m, thrust::default_random_engine& rng, glm::vec3& Li) {
    thrust::uniform_int_distribution<int> uLight(0, num_lights-1);
    const Light& light = lights[uLight(rng)];
    const Geom& lGeo = geo[light.geomid];
    const Material& lMat = materials[lGeo.materialid];

    glm::vec3 lightSamplePos = glm::vec3(lGeo.transform * glm::vec4(calculateRandomPositionOnSquare(rng), 0.0f, 1.0f));
    glm::vec3 lightSampleNorm = glm::normalize(glm::vec3(lGeo.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    // Pdf
    glm::vec3 diff = lightSamplePos - intersect;
    glm::vec3 wi = normalize(diff);
    float lightPdf = 1.0f / static_cast<float>(num_lights) * light.localPdf * dot(diff, diff) / abs(dot(lightSampleNorm, -wi));
    glm::vec3 fLambert = glm::vec3(0.0f);
    float bsdfPdf = 0.0f;
    if (m.isVolume) {
        bsdfPdf = henyeyGreensteinDouble(dot(wi, wo), m.g1, m.g2, m.gBlend, vd);
        fLambert = m.color * glm::vec3(bsdfPdf);
    }
    else {
        bsdfPdf = pdfBSDF(wi, normal, m, fLambert);
    }

    Ray shadowRay;
    shadowRay.origin = intersect + normal * BOUNCE_EPS;
    shadowRay.direction = wi;

    int hit_geom_index;
    float t_min;
    glm::vec3 shadow_intersect, shadow_normal; bool hitVolume = false;
    sceneIntersectionTest(geo, vd, shadowRay, num_geoms, rng, hit_geom_index, t_min, shadow_intersect, shadow_normal, hitVolume);
    if(hit_geom_index == light.geomid && !hitVolume) {
        Li = fLambert * glm::vec3(lMat.emittance) * powerHeuristic(lightPdf, bsdfPdf) / lightPdf;
    } else {
        Li = glm::vec3(0.0f);
    }
}