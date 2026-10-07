#include "sdf_scene.cuh"
#include "sdf_utils.cuh"
#include "sdf_structs.cuh"

using namespace glm;

__device__ float scene_intersect(Ray r, float t_min, float t_max, float eps) {
    // conservative bounding sphere
    constexpr float radius = 1.5f;
    float b = glm::dot(r.org, r.dir);
    float h = b * b - (glm::dot(r.org, r.org) - radius * radius);
    if (h < 0.f) return -1.f;

    float root = sqrtf(h);
    t_min = glm::max(t_min, -b - root);
    t_max = glm::min(t_max, -b + root);
    if (t_min > t_max)
        return -1.f;
    
    return trace_sdf<Mandelbox1E6DE>(r, Mandelbox1E6DE{}, t_min, t_max, eps);
}

__device__ vec3 scene_normal(vec3 p, float eps) {
    return calc_normal<Mandelbox1E6DE>(p, Mandelbox1E6DE{}, eps);
}
