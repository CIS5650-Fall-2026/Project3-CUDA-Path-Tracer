#include "interactions.h"
#include "intersections.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void cudaCheck(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__device__ bool nearlyEqual(float a, float b, float epsilon = 2e-5f) { return fabsf(a - b) <= epsilon; }
__device__ bool near3(glm::vec3 a, glm::vec3 b, float e = 2e-5f) {
    return nearlyEqual(a.x, b.x, e) && nearlyEqual(a.y, b.y, e) && nearlyEqual(a.z, b.z, e);
}
__device__ void check(bool condition, unsigned int bit, unsigned int& failures) {
    if (!condition) failures |= 1u << bit;
}
// This test calls the production device functions in a real CUDA kernel.
__global__ void deviceChecks(Material glass, Material standard, Material metal,
    Geom sphere, Geom cube, unsigned int* output) {
    unsigned int failures = 0;
    const glm::vec3 normal(0, 0, 1);
    const auto reflected = bsdf::sampleSmoothDielectric(glass, normal, true, 0.02f);
    const auto entered = bsdf::sampleSmoothDielectric(glass, normal, true, 0.5f);
    const auto exited = bsdf::sampleSmoothDielectric(glass, normal, false, 0.5f);
    check(reflected.delta && !reflected.transmission && nearlyEqual(reflected.branchProbability, .04f)
        && near3(reflected.deltaWeight, glm::vec3(1)), 0, failures);
    check(entered.transmission && nearlyEqual(entered.branchProbability, .96f)
        && near3(entered.deltaWeight, glm::vec3(4.0f / 9.0f)), 1, failures);
    check(exited.transmission && near3(exited.deltaWeight, glm::vec3(2.25f))
        && near3(entered.deltaWeight * exited.deltaWeight, glm::vec3(1)), 2, failures);
    const auto oblique = bsdf::sampleSmoothDielectric(glass,
        glm::vec3(.5f, 0, sqrtf(.75f)), true, .99f);
    check(oblique.transmission && nearlyEqual(oblique.wi.x, -1.0f / 3.0f)
        && nearlyEqual(glm::length(oblique.wi), 1) && oblique.wi.z < 0, 3, failures);
    const auto obliqueExit = bsdf::sampleSmoothDielectric(glass,
        glm::vec3(1.0f / 3.0f, 0, sqrtf(8.0f / 9.0f)), false, .99f);
    check(obliqueExit.transmission && nearlyEqual(obliqueExit.wi.x, -.5f), 4, failures);
    const auto tir = bsdf::sampleSmoothDielectric(glass, glm::vec3(.8f, 0, .6f), false, .99f);
    check(!tir.transmission && nearlyEqual(tir.branchProbability, 1) && near3(tir.deltaWeight, glm::vec3(1)), 5, failures);
    Material diffuse = standard; diffuse.Ks = glm::vec3(0);
    Material glossy = standard; glossy.Kd = glm::vec3(0);
    const glm::vec3 wi = glm::normalize(glm::vec3(.3f, .2f, 1));
    check(nearlyEqual(bsdf::pdf(standard, normal, wi),
        .5f * (bsdf::pdf(diffuse, normal, wi) + bsdf::pdf(glossy, normal, wi))), 6, failures);
    thrust::default_random_engine rng(987654321u);
    const auto sampled = bsdf::sample(diffuse, normal, true, rng);
    check(sampled.pdf > 0 && near3(sampled.f * sampled.wi.z / sampled.pdf, diffuse.Kd), 7, failures);
    Material oren = diffuse; oren.diffuseSigmaDegrees = 30;
    const float sigma2 = (3.14159265358979323846f / 6) * (3.14159265358979323846f / 6);
    const float A = 1 - sigma2 / (2 * (sigma2 + .33f));
    check(near3(bsdf::evaluate(oren, normal, normal, true),
        diffuse.Kd * (A / 3.14159265358979323846f)), 8, failures);
    const glm::vec3 e = metal.metal.etaT, k = metal.metal.k;
    const glm::vec3 expectedF = ((e - glm::vec3(1)) * (e - glm::vec3(1)) + k * k) /
        ((e + glm::vec3(1)) * (e + glm::vec3(1)) + k * k);
    check(near3(bsdf::evaluate(metal, normal, normal, true) *
        (4 * 3.14159265358979323846f * metal.alpha * metal.alpha), expectedF), 9, failures);
    for (int object = 0; object < 2; ++object) {
        const Geom g = object == 0 ? sphere : cube;
        Ray ray;
        ray.origin = multiplyMV(g.transform, glm::vec4(0, 0, 2, 1));
        ray.direction = glm::normalize(multiplyMV(g.transform, glm::vec4(0, 0, -1, 0)));
        glm::vec3 position, outward; bool outside = false;
        const float t = object == 0 ? sphereIntersectionTest(g, ray, position, outward, outside)
            : boxIntersectionTest(g, ray, position, outward, outside);
        const glm::vec3 expectedNormal = glm::normalize(multiplyMV(g.invTranspose, glm::vec4(0, 0, 1, 0)));
        check(t > 0 && outside && near3(outward, expectedNormal) && nearlyEqual(glm::length(outward), 1),
            10 + object * 3, failures);
        const glm::vec3 entry = position;
        const Ray incident = ray;
        ray.origin = multiplyMV(g.transform, glm::vec4(0, 0, 0, 1));
        ray.direction = glm::normalize(multiplyMV(g.transform, glm::vec4(0, 0, 1, 0)));
        const float exitT = object == 0 ? sphereIntersectionTest(g, ray, position, outward, outside)
            : boxIntersectionTest(g, ray, position, outward, outside);
        check(exitT > 0 && !outside && near3(outward, expectedNormal), 11 + object * 3, failures);
        bool offsetsValid = true, sawTransmission = false;
        for (unsigned int i = 1; i <= 32; ++i) {
            PathSegment path; path.ray = incident; path.color = glm::vec3(1);
            path.pixelIndex = 0; path.remainingBounces = 8;
            thrust::default_random_engine random(12979u + 24593u * i);
            scatterRay(path, entry, expectedNormal, glass, random);
            const float destination = glm::dot(path.ray.direction, expectedNormal);
            const float offset = glm::dot(path.ray.origin - entry, expectedNormal);
            sawTransmission = sawTransmission || destination < 0;
            offsetsValid = offsetsValid && path.remainingBounces == 7 && offset * destination > 0
                && nearlyEqual(glm::length(path.ray.direction), 1);
        }
        check(offsetsValid && sawTransmission, 12 + object * 3, failures);
    }
    *output = failures;
}
int main() {
    unsigned int* device = nullptr;
    try {
        int count = 0; cudaCheck(cudaGetDeviceCount(&count));
        if (count == 0) throw std::runtime_error("No CUDA device; device checks did not run");
        Material glass; glass.type = MaterialType::Transmission; glass.transmission.setPreset(0);
        Material standard; standard.Kd = glm::vec3(.4f, .5f, .6f); standard.Ks = glm::vec3(.3f);
        standard.setRoughness(.1f);
        Material metal; metal.type = MaterialType::Metal; metal.Kd = glm::vec3(0);
        metal.Ks = glm::vec3(1); metal.setRoughness(.1f); metal.metal.setPreset(1);
        Geom sphere{}, cube{};
        sphere.transform = glm::translate(glm::mat4(1), glm::vec3(1, 2, -1)) *
            glm::rotate(glm::mat4(1), .4f, glm::vec3(0, 1, 0)) *
            glm::scale(glm::mat4(1), glm::vec3(2, 1, 3));
        sphere.inverseTransform = glm::inverse(sphere.transform);
        sphere.invTranspose = glm::transpose(sphere.inverseTransform);
        cube = sphere;
        cudaCheck(cudaMalloc(&device, sizeof(unsigned int)));
        deviceChecks<<<1, 1>>>(glass, standard, metal, sphere, cube, device);
        cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
        unsigned int failures = 0;
        cudaCheck(cudaMemcpy(&failures, device, sizeof(failures), cudaMemcpyDeviceToHost));
        cudaCheck(cudaFree(device)); device = nullptr;
        const char* names[] = {"normal reflection/Fresnel", "entry weight", "exit/roundtrip weight",
            "entry Snell direction", "exit Snell direction", "total internal reflection",
            "mixture PDF", "Lambert throughput", "Oren-Nayar normal value", "conductor normal Fresnel",
            "sphere entry normal", "sphere exit normal", "sphere offsets", "cube entry normal",
            "cube exit normal", "cube offsets"};
        for (unsigned int i = 0; i < 16; ++i)
            std::cout << ((failures & (1u << i)) ? "FAIL: " : "PASS: ") << names[i] << " (CUDA device)\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        if (device) cudaFree(device);
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
