#include "sdf_structs.cuh"
#include <glm/geometric.hpp>

using namespace glm;

__device__ float SphereSDF::operator()(vec3 p) const {
    const float r = 1.f;
    return length(p) - r;
}

__device__ inline void double_angle(float& s, float& c) {
    float s2 = 2.f * s * c;
    float c2 = c * c - s * s;

    s = s2;
    c = c2;
}

__device__ __forceinline__ glm::vec3 calc_w(glm::vec3 w, const glm::vec3& p) {
    float x = w.x;
    float y = w.y;
    float z = w.z;

    float r2 = x * x + y * y + z * z;
    if (r2 <= 1e-20f)
        return p;

    float inv_r = rsqrtf(r2);

    // theta = acos(y / r)
    float cos_theta = y * inv_r;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    // phi = atan2(x, z)
    float rho2 = x * x + z * z;

    float sin_phi;
    float cos_phi;
    if (rho2 > 0.00001f) {
        float inv_rho = rsqrtf(rho2);
        sin_phi = x * inv_rho;
        cos_phi = z * inv_rho;
    } else {
        // undefined at the pole
        sin_phi = 0.0f;
        cos_phi = 1.0f;
    }

    // 8theta
    double_angle(sin_theta, cos_theta);
    double_angle(sin_theta, cos_theta);
    double_angle(sin_theta, cos_theta);

    // 8phi
    double_angle(sin_phi, cos_phi);
    double_angle(sin_phi, cos_phi);
    double_angle(sin_phi, cos_phi);

    float r4 = r2 * r2;
    float r8 = r4 * r4;

    return p + r8 * glm::vec3(sin_theta * sin_phi, cos_theta, sin_theta * cos_phi);
}

__device__ float MandelbulbDE::operator()(vec3 p) const {
    constexpr int iterations = 1;
    vec3 w = p;
    float m = dot(w, w);

    // vec4 trap = vec4(abs(w), m);
    float dz = 1.f;

    for (int i = 0; i < iterations; ++i) {
        // trigonometric version (MUCH faster than polynomial)

        // dz = 8*z^7*dz
        dz = 8.f * __powf(m, 3.5f) * dz + 1.f;

        // z = z^8+c
        w = calc_w(w, p);

        // trap = min(trap, vec4(abs(w), m));

        m = dot(w, w);
        if (m > 256.f)
            break;
    }

    // resColor = vec4(m, trap.yzw);

    // distance estimation (through the Hubbard-Douady potential)
    return 0.25f * __logf(m) * sqrtf(m) / dz;
}


// https://www.shadertoy.com/view/7tfSzB
// Mandelbox variant with smooth box fold, rounded-box orbit trap
namespace mbox {
constexpr float fixed_radius2 = 1.9f;
constexpr float min_radius2 = 0.5f;
constexpr float folding_limit = 1.f;
constexpr float scale = -2.8f;
constexpr float k = 0.05f;
constexpr float zoom = 0.3f;

// polynomial smooth min
__device__ __forceinline__ float pmin(float a, float b, float k) {
    float h = __saturatef(0.5f + 0.5f * __fdividef(b - a, k));
    return fmaf(h, a - b, b) - k * h * (1.f - h);
}

__device__ __forceinline__ void sphere_fold(vec3& z, float& dz) {
    float r2 = dot(z, z);
    if (r2 < min_radius2) {
        constexpr float temp = fixed_radius2 / min_radius2;
        z *= temp;
        dz *= temp;
    } else if (r2 < fixed_radius2) {
        float temp = __fdividef(fixed_radius2, r2);
        z *= temp;
        dz *= temp;
    }
}

// smooth version of z = clamp(z, -limit, limit) * 2 - z
__device__ __forceinline__ void box_fold(float k, vec3& z) {
    vec3 zz(copysignf(pmin(fabsf(z.x), folding_limit, k), z.x),
            copysignf(pmin(fabsf(z.y), folding_limit, k), z.y),
            copysignf(pmin(fabsf(z.z), folding_limit, k), z.z));
    z = zz * 2.f - z;
}

__device__ __forceinline__ float sphere(vec3 p, float r) {
    return length(p) - r;
}

// box frame: one axis of the outer box, the other two of the inset edges
__device__ __forceinline__ float boxf_axis(float px, float qy, float qz) {
    float ax = fmaxf(px, 0.f);
    float ay = fmaxf(qy, 0.f);
    float az = fmaxf(qz, 0.f);
    return sqrtf(ax * ax + ay * ay + az * az) + fminf(fmaxf(px, fmaxf(qy, qz)), 0.f);
}

__device__ __forceinline__ float boxf(vec3 p, float b, float e) {
    p = vec3(fabsf(p.x), fabsf(p.y), fabsf(p.z)) - b;
    vec3 q = vec3(fabsf(p.x + e), fabsf(p.y + e), fabsf(p.z + e)) - e;
    return fminf(fminf(boxf_axis(p.x, q.y, q.z), boxf_axis(p.y, q.x, q.z)),
                 boxf_axis(p.z, q.x, q.y));
}

__device__ __forceinline__ float mb(vec3 z) {
    vec3 offset = z;
    float dr = 1.f;
    float fd = 0.f;

#pragma unroll
    for (int n = 0; n < 5; ++n) {
        box_fold(__fdividef(k, dr), z);
        sphere_fold(z, dr);
        z = scale * z + offset;
        dr = fmaf(dr, fabsf(scale), 1.f);
        float r = n < 4 ? boxf(z, 5.f, 0.5f) : sphere(z, 5.f);
        float dd = __fdividef(r, dr); // dr is always positive
        if (n < 3 || dd < fd)
            fd = dd;
    }
    return fd;
}
} // namespace mbox

__device__ float Mandelbox1E6DE::operator()(vec3 p) const {
    // extent is ~4.2 before zoom, so ~1.3 after: fits the 1.5 bounding sphere
    constexpr float inv_zoom = 1.f / mbox::zoom;
    return mbox::mb(p * inv_zoom) * mbox::zoom;
}
