#pragma once

#include <cuda/std/variant>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct Lambertian {
    glm::vec3 color;
    __device__ bool is_delta() const {
        return false;
    }
};
struct Conductor {
    glm::vec3 eta;
    glm::vec3 k;
    float roughness;
    float anisotropy;
    __device__ bool is_delta() const {
        return roughness == 0.f;
    }
};
struct Dielectric {
    float ior;
    float roughness;
    __device__ bool is_delta() const {
        return roughness == 0.f;
    }
};

struct DisneyDiffuse {
    glm::vec3 color;
    float roughness;
    float subsurface;
    __device__ bool is_delta() const {
        return false;
    }
};

struct DisneyMetal {
    glm::vec3 color;
    glm::vec3 edge_tint;
    float roughness;
    float anisotropic;
    __device__ bool is_delta() const {
        return roughness == 0.f;
    }
};

struct DisneyClearcoat {
    float gloss;
    __device__ bool is_delta() const {
        return false;
    }
};

struct DisneyGlass {
    glm::vec3 color;
    float roughness;
    float anisotropic;
    float ior;
    __device__ bool is_delta() const {
        return roughness == 0.f;
    }
};

struct DisneySheen {
    glm::vec3 color;
    float sheen_tint;
    __device__ bool is_delta() const {
        return false;
    }
};

struct DisneyBsdf {
    glm::vec3 color;
    float specular_transmission;
    float metallic;
    glm::vec3 edge_tint;
    float subsurface;
    float specular;
    float roughness;
    float specular_tint;
    float anisotropic;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    float ior;
    __device__ bool is_delta() const {
        return false;
    }
};

using BsdfVariant =
    cuda::std::variant<Lambertian, Conductor, Dielectric, DisneyDiffuse, DisneyMetal,
                       DisneyClearcoat, DisneyGlass, DisneySheen, DisneyBsdf>;
