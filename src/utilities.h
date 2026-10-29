#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>

constexpr float PI = 3.1415926535897932384626422832795028841971f;
constexpr float TWO_PI = 6.2831853071795864769252867665590057683943f;
// Ray origin offset off a surface, scaled with the hit point's magnitude
constexpr float EPSILON = 0.00001f;

#ifdef __CUDACC__
__host__ __device__
#endif
inline float maxComponent(glm::vec3 v)
{
    return glm::max(v.x, glm::max(v.y, v.z));
}

// Rec. 709 luminance of a linear color
#ifdef __CUDACC__
__host__ __device__
#endif
inline float luminance(glm::vec3 c)
{
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

// Here rather than in sampling.h so the OptiX programs, which cannot include thrust, can hash too.
#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// With the dot ("Duck.GLB" -> ".glb"); empty when there is none.
std::string lowercaseExtension(const std::string& path);

[[noreturn]] void fatal(const char* format, ...);

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)
void cudaCheck(cudaError_t result, const char* call, const char* file, int line);

// T * R * S with Euler degrees as Rx * Ry * Rz, so a point turns about z first.
glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
