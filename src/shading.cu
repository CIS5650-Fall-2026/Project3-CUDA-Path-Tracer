#include "bsdf/bsdf.cuh"
#include "bsdf/bxdf_utils.cuh"
#include "config.h"
#include "glm/ext/matrix_float3x3.hpp"
#include "intersections.cuh"
#include "light_sampling.cuh"
#include "math_utils.h"
#include "render_settings.cuh"
#include "sampling.cuh"
#include "scene_structs.h"
#include "shading.cuh"

__device__ __forceinline__ float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

__device__ __forceinline__ void terminate_path(PathSegment& path) {
    path.remaining_bounces = 0;
}

__device__ glm::vec3 eval_environment(const cudaTextureObject_t& env_texture, glm::vec3 wi) {
    if (env_texture == 0) {
        return glm::vec3(0.0f);
    }

    glm::vec3 d = glm::normalize(wi);

    float theta = acosf(glm::clamp(d.y, -1.f, 1.f));
    float phi = atan2f(d.z, d.x);
    if (phi < 0.f) {
        phi += 2.f * PI;
    }

    float u = phi / (2.f * PI);
    float v = theta / PI;

    float4 rgb = tex2D<float4>(env_texture, u, v);
    return c_settings.env_strength * glm::vec3(rgb.x, rgb.y, rgb.z);
}

__device__ cstd::optional<ShadowRay>
estimate_direct_lighting(const PathSegment& path, glm::vec3 p, glm::vec3 nor,
                         const glm::mat3& to_world, glm::vec3 wo, const Material& m, RngEng& rng,
                         const LightSampler& s, const Geom* geoms) {
    cstd::optional<LightSample> ls = sample_direct_light(p, nor, s, geoms, rng);

    if (!ls || ls->pdf == 0.f) {
        return cstd::nullopt;
    }

    BsdfEval be = eval_pdf_bsdf(wo, ls->wi * to_world, m);
    if (be.f == glm::vec3(0.f)) {
        return cstd::nullopt;
    }

    bool is_env = ls->light_idx < 0;
    glm::vec3 li =
        is_env ? eval_environment(s.env_texture, ls->wi) : s.lights[ls->light_idx].emission.radiance();

    float w = path.remaining_bounces > 1 ? power_heuristic(ls->pdf, be.pdf) : 1.f;

    ShadowRay sray{};
    sray.ray = bx::spawn_ray(p, ls->wi, nor);
    sray.pixel_index = path.pixel_index;
    // so we don't intersect with the same light when tracing shadow rays
    if (is_env) {
        sray.t_max = FLT_MAX;
    } else {
        glm::vec3 light_p = p + ls->wi * ls->dist;
        sray.t_max = glm::dot(light_p - sray.ray.org, ls->wi) * (1.f - numeric::shadow_margin);
    }

    sray.contribution = path.throughput * li * be.f * w / ls->pdf;

    return sray;
}

__device__ void scatter_ray(PathSegment& path, glm::vec3 p, glm::vec3 nor,
                            const glm::mat3& to_world, glm::vec3 wo, const Material& m,
                            RngEng& rng) {
    BsdfSample bs = sample_bsdf(wo, m, rng);
    if (bs.type == BxdfFlag::Unset || bs.pdf == 0.f) {
        path.throughput = glm::vec3(0.f);
        terminate_path(path);
        return;
    }

    path.throughput *= bs.f / bs.pdf;
    path.prev_was_delta = (bs.type & BxdfFlag::Specular) != BxdfFlag::Unset;
    path.prev_bsdf_pdf = bs.pdf;
    path.ray = bx::spawn_ray(p, to_world * bs.wi, nor);
    path.remaining_bounces--;
}

__global__ void shade_material(int iter, int num_paths, int depth,
                               const ShadeableIntersection* shadeable_intersections,
                               const MatId* isect_mat_ids, PathSegment* path_segments,
                               ShadowRay* shadow_rays, const Material* materials,
                               LightSampler light_sampler, const Geom* geoms, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
    shadow_rays[idx].pixel_index = -1;
    if (path_segments[idx].remaining_bounces <= 0) {
        return;
    }

    // miss / environment

    PathSegment path = path_segments[idx];
    MatId mat_id = isect_mat_ids[idx];
    if (mat_id == MAT_MISS) {
        glm::vec3 le = eval_environment(light_sampler.env_texture, path.ray.dir);
        float w = 1.f;

#if LI_MIS
        if (depth > 0 && !path.prev_was_delta && light_sampler.p_env > 0.f) {
            float li_pdf = pmf_env(light_sampler) * pdf_env_light(path.ray.dir);
            w = power_heuristic(path.prev_bsdf_pdf, li_pdf);
        }
#endif
        image[path.pixel_index] += path.throughput * le * w;
        terminate_path(path_segments[idx]);
        return;
    }

    // emissive

    if (mat_id == MAT_LIGHT) {
        const Light& light = light_sampler.lights[shadeable_intersections[idx].light_idx];
#if LI_MIS
        if (depth == 0 || path.prev_was_delta) {
            image[path.pixel_index] += path.throughput * light.emission.radiance();
        } else {
            // light sampling
            float li_pdf = pmf_area(light_sampler) * pdf_area_light(path.ray, light, geoms);

            float w = power_heuristic(path.prev_bsdf_pdf, li_pdf);
            image[path.pixel_index] += w * path.throughput * light.emission.radiance();
        }
#else
        image[path.pixel_index] += path.throughput * light.emission.radiance();
#endif
        terminate_path(path_segments[idx]);
        return;
    }

    // bounce
    Material material = materials[mat_id];
    if (is_emissive(material)) {
        image[path.pixel_index] += path.throughput * material.emission.radiance();
    }

    ShadeableIntersection intersection = shadeable_intersections[idx];
    RngEng rng = make_seeded_rng(iter, idx, depth);
    glm::vec3 p = get_point_on_ray(path.ray, intersection.t);
    const glm::vec3& nor = intersection.surface_normal;
    glm::mat3 to_world = bx::local_to_world(nor);
    glm::vec3 wo = -path.ray.dir * to_world;

#if LI_MIS
    if (!is_delta(material)) {
        // if not delta (so change this when I add microfacet)
        cstd::optional<ShadowRay> sray = estimate_direct_lighting(
            path, p, nor, to_world, wo, material, rng, light_sampler, geoms);

        if (sray) {
            shadow_rays[idx] = *sray;
        }
    }

    scatter_ray(path, p, nor, to_world, wo, material, rng);
#else
    scatter_ray(path, p, nor, to_world, wo, material, rng);
#endif

#if RUSSIAN_ROULETTE
    if (depth > 3 && path.remaining_bounces > 0) {
        UnifDist<float> u01(0, 1);

        float survive_p = max(path.throughput.x, max(path.throughput.y, path.throughput.z));
        survive_p = glm::clamp(survive_p, 0.05f, 0.95f);
        if (u01(rng) > survive_p) {
            terminate_path(path_segments[idx]);
            return;
        } else {
            path.throughput /= survive_p;
        }
    }
#endif
    path_segments[idx] = path;
}

__global__ void trace_shadow_rays(int num_srays, int num_geoms, ShadowRay* shadow_rays,
                                  const Geom* geoms, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_srays || shadow_rays[idx].pixel_index == -1) {
        return;
    }

    ShadowRay sray = shadow_rays[idx];
    if (visible_to_light(sray.ray, sray.t_max, geoms, num_geoms)) {
        image[sray.pixel_index] += sray.contribution;
    }
}
