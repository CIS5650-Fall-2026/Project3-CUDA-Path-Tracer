#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ glm::vec3 fresnel_conductor_eval(float cos_theta_i, const glm::vec3& eta_i,
                                            const glm::vec3& eta_t, const glm::vec3& k) {
    cos_theta_i = glm::clamp(fabsf(cos_theta_i), 0.f, 1.f);

    glm::vec3 eta = eta_t / eta_i;
    glm::vec3 eta_k = k / eta_i;

    float cos2_theta = cos_theta_i * cos_theta_i;
    float sin2_theta = 1.f - cos2_theta;
    float sin4_theta = sin2_theta * sin2_theta;

    glm::vec3 eta2 = eta * eta;
    glm::vec3 k2 = eta_k * eta_k;

    glm::vec3 a2_plus_b2 =
        glm::sqrt((eta2 - k2 - sin2_theta) * (eta2 - k2 - sin2_theta) + 4.f * eta2 * k2);

    glm::vec3 a = glm::sqrt(0.5f * (a2_plus_b2 + eta2 - k2 - sin2_theta));

    glm::vec3 r_perp = (a2_plus_b2 - 2.f * a * cos_theta_i + cos2_theta) /
                       (a2_plus_b2 + 2.f * a * cos_theta_i + cos2_theta);

    glm::vec3 r_parallel =
        r_perp * (cos2_theta * a2_plus_b2 - 2.f * a * cos_theta_i * sin2_theta + sin4_theta) /
        (cos2_theta * a2_plus_b2 + 2.f * a * cos_theta_i * sin2_theta + sin4_theta);

    return 0.5f * (r_perp + r_parallel);
}

__device__ __forceinline__ float fresnel_dielectric_eval(float cos_theta_i, float eta_i, float eta_t) {
    cos_theta_i = glm::clamp(cos_theta_i, -1.f, 1.f);
    bool entering = cos_theta_i > 0.f;
    if (!entering) {
        float tmp = eta_i;
        eta_i = eta_t;
        eta_t = tmp;
        cos_theta_i = fabsf(cos_theta_i);
    }

    float sin_theta_i = sqrtf(fmaxf(0.f, 1.f - cos_theta_i * cos_theta_i));
    float sin_theta_t = eta_i / eta_t * sin_theta_i;
    if (sin_theta_t >= 1.f) {
        // total internal reflection
        return 1.f;
    }
    float cos_theta_t = sqrtf(fmaxf(0.f, 1.f - sin_theta_t * sin_theta_t));

    float r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) /
                   ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    float r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) /
                   ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parl * r_parl + r_perp * r_perp) / 2.f;
}
