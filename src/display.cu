#include "display.cuh"

__host__ __device__ glm::vec3 pow_components(glm::vec3 base, glm::vec3 exponent) {
    return glm::vec3(powf(base.x, exponent.x), powf(base.y, exponent.y), powf(base.z, exponent.z));
}

#if !defined(PURE_GAMMA)
#define SRGB_OETF
#endif

#if defined(PURE_GAMMA)
__host__ __device__ glm::vec3 to_linear(glm::vec3 sRGB) {
    return pow_components(sRGB, glm::vec3(2.2f));
}

__host__ __device__ glm::vec3 from_linear(glm::vec3 linearRGB) {
    return pow_components(linearRGB, glm::vec3(1.f / 2.2f));
}
#elif defined(SRGB_OETF)
__host__ __device__ glm::vec3 to_linear(glm::vec3 sRGB) {
    glm::bvec3 cutoff = glm::lessThan(sRGB, glm::vec3(0.04045f));
    glm::vec3 higher = pow_components((sRGB + glm::vec3(0.055f)) / 1.055f, glm::vec3(2.4f));
    glm::vec3 lower = sRGB / 12.92f;
    return glm::mix(higher, lower, cutoff);
}

__host__ __device__ glm::vec3 from_linear(glm::vec3 linearRGB) {
    glm::bvec3 cutoff = glm::lessThan(linearRGB, glm::vec3(0.0031308f));
    glm::vec3 higher =
        1.055f * pow_components(linearRGB, glm::vec3(1.f / 2.4f)) - glm::vec3(0.055f);
    glm::vec3 lower = linearRGB * 12.92f;
    return glm::mix(higher, lower, cutoff);
}
#endif

__host__ __device__ glm::vec3 saturate(glm::vec3 v) {
    return glm::clamp(v, 0.f, 1.f);
}

__host__ __device__ glm::vec3 agx_curve3(glm::vec3 v) {
    const float threshold = 0.6060606060606061f;
    const float a_up = 69.86278913545539f;
    const float a_down = 59.507875f;
    const float b_up = 13.f / 4.f;
    const float b_down = 3.f;
    const float c_up = -4.f / 13.f;
    const float c_down = -1.f / 3.f;

    glm::vec3 mask = glm::step(v, glm::vec3(threshold));
    glm::vec3 a = glm::vec3(a_up) + (a_down - a_up) * mask;
    glm::vec3 b = glm::vec3(b_up) + (b_down - b_up) * mask;
    glm::vec3 c = glm::vec3(c_up) + (c_down - c_up) * mask;
    return glm::vec3(0.5f) +
           (glm::vec3(-2.f * threshold) + 2.f * v) *
               pow_components(
                   glm::vec3(1.f) + a * pow_components(glm::abs(v - glm::vec3(threshold)), b), c);
}

__host__ __device__ glm::vec3 agx_tonemapping(glm::vec3 ci) {
    const float min_ev = -12.473931188332413f;
    const float max_ev = 4.026068811667588f;
    const float dynamic_range = max_ev - min_ev;

    const glm::mat3 agx_mat(0.8424010709504686f, 0.04240107095046854f, 0.04240107095046854f,
                            0.07843650156180276f, 0.8784365015618028f, 0.07843650156180276f,
                            0.0791624274877287f, 0.0791624274877287f, 0.8791624274877287f);
    const glm::mat3 agx_mat_inv(1.1969986613119143f, -0.053001338688085674f, -0.053001338688085674f,
                                -0.09804562695225345f, 1.1519543730477466f, -0.09804562695225345f,
                                -0.09895303435966087f, -0.09895303435966087f, 1.151046965640339f);

    ci = agx_mat * ci;
    glm::vec3 log_ci(log2f(ci.x), log2f(ci.y), log2f(ci.z));
    glm::vec3 ct = saturate(log_ci * (1.f / dynamic_range) - glm::vec3(min_ev / dynamic_range));
    glm::vec3 co = agx_curve3(ct);
    co = agx_mat_inv * co;
    return co;
}

__host__ __device__ glm::ivec3 to_display(glm::vec3 linear_rgb, bool agx_tonemap) {
    glm::vec3 color =
        agx_tonemap ? saturate(agx_tonemapping(linear_rgb)) : from_linear(saturate(linear_rgb));

    glm::ivec3 icolor = glm::clamp(glm::ivec3(color * 255.f), 0, 255);
    return icolor;
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void send_image_to_pbo(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool agx_tonemap) {
    if (iter == 0) {
        return;
    }

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 color = image[index] / static_cast<float>(iter);
        glm::ivec3 icolor = to_display(color, agx_tonemap);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = icolor.x;
        pbo[index].y = icolor.y;
        pbo[index].z = icolor.z;
    }
}
