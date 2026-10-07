#include "config.h"
#include "intersections.cuh"
#include "light_sampling.cuh"
#include "sampling.cuh"

__device__ float area_to_solid_angle_pdf(float pdf_area, float dist2, float cos_light) {
    return dist2 * pdf_area / cos_light;
}

__device__ cstd::optional<LightSample> sample_plane_light(glm::vec3 p, const Geom& plane,
                                                          RngEng& rng) {
    LightSample sample{};
    UnifDist<float> u01(0, 1);

    float surface_area = plane.transform.scale.x * plane.transform.scale.z;
    glm::vec2 xi(u01(rng) - 0.5f, u01(rng) - 0.5f);
    glm::vec3 light_pos = multiply_mv(plane.transform.matrix, glm::vec4(xi.x, 0, xi.y, 1));
    glm::vec3 light_normal =
        glm::normalize(multiply_mv(plane.transform.inv_transpose, glm::vec4(0, 1, 0, 0)));

    glm::vec3 v = light_pos - p;
    sample.dist = length(v);
    if (sample.dist == 0.f) {
        return cstd::nullopt;
    }
    sample.wi = v / sample.dist;

    float cos_t = glm::dot(-sample.wi, light_normal);
    if (cos_t <= numeric::min_cos) {
        return cstd::nullopt;
    }

    sample.pdf = sample.dist * sample.dist / (cos_t * surface_area);
    return sample;
}

__device__ float pdf_plane_light(Ray r, const Geom& plane) {
    glm::vec3 hit_pt;
    bool outside;
    float t = plane_intersection_test(plane, r, &hit_pt, nullptr, &outside);

    if (t <= 0.f || outside) {
        return 0.f;
    }

    glm::vec3 light_normal =
        glm::normalize(multiply_mv(plane.transform.inv_transpose, glm::vec4(0, 1, 0, 0)));

    float surface_area = plane.transform.scale.x * plane.transform.scale.z;
    float dist2 = glm::dot(hit_pt - r.org, hit_pt - r.org);
    float cos_light = glm::dot(-r.dir, light_normal);
    if (cos_light <= numeric::min_cos) {
        return 0.f;
    }
    return area_to_solid_angle_pdf(1.f / surface_area, dist2, cos_light);
}

__device__ float pdf_env_light(glm::vec3 wi) {
    return square_to_sphere_uniform_pdf(wi);
}

__device__ LightSample sample_env_light(RngEng& rng) {
    UnifDist<float> u01(0.0f, 1.0f);

    glm::vec2 xi(u01(rng), u01(rng));

    LightSample sample{};
    sample.wi = square_to_sphere_uniform(xi);
    sample.pdf = pdf_env_light(sample.wi);
    sample.dist = FLT_MAX;

    return sample;
}

__device__ float pdf_area_light(const Ray& r, const Light& light, const Geom* geoms) {
    const Geom& geom = geoms[light.geom_id];
    switch (geom.type) {
    case GeomType::Plane:
        return pdf_plane_light(r, geom);
    default:
        return 0.f;
    }
}

__device__ cstd::optional<LightSample> sample_area_light(glm::vec3 p, const Light& light,
                                                         const Geom* geoms, RngEng& rng) {
    const Geom& geom = geoms[light.geom_id];
    switch (geom.type) {
    case GeomType::Plane:
        return sample_plane_light(p, geom, rng);
    default:
        return cstd::nullopt;
    }
}

__device__ cstd::optional<LightSample> sample_direct_light(glm::vec3 p, glm::vec3 nor,
                                                           const LightSampler& s,
                                                           const Geom* geoms, RngEng& rng) {
    if (s.p_env == 0.f && s.num_lights == 0) {
        return cstd::nullopt;
    }
    UnifDist<float> u01(0, 1);
    float u = u01(rng);

    cstd::optional<LightSample> sample;
    if (s.num_lights == 0 || u < s.p_env) {
        sample = sample_env_light(rng);
        sample->pdf *= pmf_env(s);
        sample->light_idx = -1;
    } else {
        float u_area = (u - s.p_env) / (1.f - s.p_env);
        int light_idx = glm::min(static_cast<int>(u_area * s.num_lights), s.num_lights - 1);

        sample = sample_area_light(p, s.lights[light_idx], geoms, rng);
        if (!sample) {
            return cstd::nullopt;
        }
        sample->pdf *= pmf_area(s);
        sample->light_idx = light_idx;
    }

    if (glm::dot(sample->wi, nor) <= 0.f) {
        return cstd::nullopt;
    }
    return sample;
}
