#include "cuda-utilities.h"
#include "utilities.h"

__host__ __device__ glm::mat3 FrameFromZ(glm::vec3 v1) {
    float sign = std::copysign(1.0f, v1.z);
    float a = -1.0f / (sign + v1.z);
    float b = v1.x * v1.y * a;
    glm::vec3 v2 = glm::vec3(1 + sign * SQR(v1.x) * a, sign * b, -sign * v1.x);
    glm::vec3 v3 = glm::vec3(b, sign + SQR(v1.y) * a, -v1.y);

    return glm::mat3(v1, v2, v3);
}

__host__ __device__ glm::vec3 SphericalDirection(float cosTheta, float sinTheta, float phi) {
    return glm::vec3(sin(phi) * cosTheta, sin(phi) * sinTheta, cos(phi));
}