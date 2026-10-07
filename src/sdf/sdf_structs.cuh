#pragma once

#include <cuda_runtime.h>
#include <glm/vec3.hpp>

struct SphereSDF {
    __device__ float operator()(glm::vec3 p) const;
};

struct MandelbulbDE {
    __device__ float operator()(glm::vec3 p) const;
};

struct Mandelbox1E6DE {
    __device__ float operator()(glm::vec3 p) const;
};
