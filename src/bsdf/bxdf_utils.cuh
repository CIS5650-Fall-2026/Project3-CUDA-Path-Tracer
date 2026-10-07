#pragma once

#include "../config.h"
#include "../scene_structs.h"

// bxdf utils
namespace bx {
static __device__ __forceinline__ float cos_theta(const glm::vec3& w) {
    return w.z;
}
static __device__ __forceinline__ float cos2_theta(const glm::vec3& w) {
    return w.z * w.z;
}
static __device__ __forceinline__ float sin2_theta(const glm::vec3& w) {
    return fmaxf(0.f, 1.f - cos2_theta(w));
}
static __device__ __forceinline__ float sin_theta(const glm::vec3& w) {
    return sqrtf(sin2_theta(w));
}
static __device__ __forceinline__ float tan_theta(const glm::vec3& w) {
    return sin_theta(w) / cos_theta(w);
}
static __device__ __forceinline__ float tan2_theta(const glm::vec3& w) {
    return sin2_theta(w) / cos2_theta(w);
}
static __device__ __forceinline__ float cos_phi(const glm::vec3& w) {
    float sin_theta_value = sin_theta(w);
    return (sin_theta_value == 0.f) ? 0.f : glm::clamp(w.x / sin_theta_value, -1.f, 1.f);
}
static __device__ __forceinline__ float sin_phi(const glm::vec3& w) {
    float sin_theta_value = sin_theta(w);
    return (sin_theta_value == 0.f) ? 0.f : glm::clamp(w.y / sin_theta_value, -1.f, 1.f);
}
static __device__ __forceinline__ float cos2_phi(const glm::vec3& w) {
    return cos_phi(w) * cos_phi(w);
}
static __device__ __forceinline__ float sin2_phi(const glm::vec3& w) {
    return sin_phi(w) * sin_phi(w);
}
// difference of azimuth angles between two vectors
static __device__ __forceinline__ float cos_d_phi(const glm::vec3& wa, const glm::vec3& wb) {
    return glm::clamp((wa.x * wb.x + wa.y * wb.y) /
                          glm::sqrt((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)),
                      -1.f, 1.f);
}
static __device__ __forceinline__ float abs_dot(const glm::vec3& w, const glm::vec3& n) {
    return fabsf(glm::dot(w, n));
}
static __device__ __forceinline__ float abs_cos(const glm::vec3& w) {
    return fabsf(cos_theta(w));
}
static __device__ __forceinline__ float non_neg_cos(const glm::vec3& w) {
    return fmaxf(0.f, cos_theta(w));
}
static __device__ __forceinline__ glm::vec3 face_forward(const glm::vec3& n, const glm::vec3& v) {
    return glm::dot(n, v) < 0.f ? -n : n;
}
static __device__ void coordinate_system(glm::vec3 in_nor, glm::vec3& out_tan, glm::vec3& out_bit) {
    glm::vec3 up = fabsf(in_nor.y) < 0.999999f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    out_tan = glm::normalize(glm::cross(up, in_nor));
    out_bit = glm::cross(in_nor, out_tan);
}

static __device__ glm::mat3 local_to_world(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinate_system(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

static __device__ glm::mat3 world_to_local(glm::vec3 nor) {
    return glm::transpose(local_to_world(nor));
}

static __device__ bool refract(glm::vec3 wo, glm::vec3 n, float eta, glm::vec3& wt) {
    // Compute cos theta using Snell's law
    float cos_theta_i = glm::dot(n, wo);
    float sin2_theta_i = fmaxf(0.f, 1.f - cos_theta_i * cos_theta_i);
    float sin2_theta_t = eta * eta * sin2_theta_i;

    // Handle total internal reflection for transmission
    if (sin2_theta_t >= 1.f)
        return false;
    float cos_theta_t = sqrtf(1.f - sin2_theta_t);
    wt = eta * -wo + (eta * cos_theta_i - cos_theta_t) * n;
    return true;
}

static __device__ Ray spawn_ray(glm::vec3 pos, glm::vec3 wi, glm::vec3 nor) {
    float side = glm::dot(wi, nor) >= 0.f ? 1.f : -1.f;
    return Ray{pos + side * numeric::ray_offset * nor, wi};
}
} // namespace bx
