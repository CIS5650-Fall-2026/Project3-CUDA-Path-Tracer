#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "interactions.cu"

constexpr int samples = 100000;

struct Result
{
    int reflected;
    int transmitted;
    int errors;
};

void checkCuda(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

__device__ bool closeVector(glm::vec3 a, glm::vec3 b)
{
    return glm::length(a - b) < 0.0001f;
}

__global__ void testGlass(Result* results)
{
    const int test = threadIdx.x;
    if (test >= 4)
    {
        return;
    }

    Material glass{};
    glass.color = glm::vec3(1.0f);
    glass.hasRefractive = 1.0f;
    glass.indexOfRefraction = 1.5f;

    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    glm::vec3 incoming;
    glm::vec3 transmittedDirection(0.0f);
    float transmissionWeight = 1.0f;

    if (test == 0)
    {
        incoming = glm::vec3(0.0f, 0.0f, -1.0f);
        transmittedDirection = incoming;
        transmissionWeight = 4.0f / 9.0f;
    }
    else if (test == 1)
    {
        // Air to glass, 30 degrees: sin(thetaT) = 1/3.
        incoming = glm::vec3(0.5f, 0.0f, -sqrtf(0.75f));
        transmittedDirection =
            glm::vec3(1.0f / 3.0f, 0.0f, -sqrtf(8.0f / 9.0f));
        transmissionWeight = 4.0f / 9.0f;
    }
    else if (test == 2)
    {
        // Glass to air, 30 degrees: sin(thetaT) = 0.75.
        incoming = glm::vec3(0.5f, 0.0f, sqrtf(0.75f));
        transmittedDirection =
            glm::vec3(0.75f, 0.0f, sqrtf(1.0f - 0.75f * 0.75f));
        transmissionWeight = 2.25f;
    }
    else
    {
        // Glass to air, 50 degrees: total internal reflection.
        const float angle = 50.0f * 3.14159265358979323846f / 180.0f;
        incoming = glm::vec3(sinf(angle), 0.0f, cosf(angle));
    }

    const glm::vec3 reflectedDirection(
        incoming.x, incoming.y, -incoming.z);

    thrust::default_random_engine rng(12345 + test * 7919);
    rng.discard(32);

    Result result{};

    for (int i = 0; i < samples; ++i)
    {
        PathSegment path{};
        path.ray.direction = incoming;
        path.color = glm::vec3(1.0f);
        path.remainingBounces = 8;

        scatterRay(path, glm::vec3(0.0f), normal, glass, rng);

        const bool reflected =
            path.ray.direction.z * incoming.z < 0.0f;

        bool valid;

        if (reflected)
        {
            ++result.reflected;
            valid =
                closeVector(path.ray.direction, reflectedDirection) &&
                closeVector(path.color, glm::vec3(1.0f));
        }
        else
        {
            ++result.transmitted;
            valid =
                test != 3 &&
                closeVector(path.ray.direction, transmittedDirection) &&
                closeVector(path.color, glm::vec3(transmissionWeight));
        }

        // The new origin must lie on the outgoing side of the surface.
        valid = valid &&
            path.ray.origin.z * path.ray.direction.z > 0.0f &&
            path.remainingBounces == 7;

        if (!valid)
        {
            ++result.errors;
        }
    }

    results[test] = result;
}

int main()
{
    Result* deviceResults = nullptr;
    checkCuda(cudaMalloc(
        reinterpret_cast<void**>(&deviceResults), 4 * sizeof(Result)));

    testGlass<<<1, 4>>>(deviceResults);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    Result results[4]{};
    checkCuda(cudaMemcpy(
        results, deviceResults, sizeof(results), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(deviceResults));

    const char* names[] = {
        "Normal incidence + Fresnel",
        "Air to glass at 30 degrees",
        "Glass to air at 30 degrees",
        "Total internal reflection at 50 degrees"
    };

    bool allPassed = true;

    for (int i = 0; i < 4; ++i)
    {
        const Result& r = results[i];
        const double reflectionRate =
            static_cast<double>(r.reflected) / samples;

        bool passed =
            r.errors == 0 &&
            r.reflected + r.transmitted == samples;

        if (i == 0)
        {
            // Allow statistical variation around the expected 4%.
            passed = passed && std::fabs(reflectionRate - 0.04) < 0.003;
        }

        if (i < 3)
        {
            passed = passed && r.reflected > 0 && r.transmitted > 0;
        }
        else
        {
            passed = passed &&
                r.reflected == samples && r.transmitted == 0;
        }

        printf(
            "%s: %s | reflected=%.3f%% | transmitted=%d | errors=%d\n",
            names[i], passed ? "PASS" : "FAIL",
            100.0 * reflectionRate, r.transmitted, r.errors);

        allPassed = allPassed && passed;
    }

    printf("%s\n", allPassed
        ? "ALL GLASS SCATTER TESTS PASSED"
        : "GLASS SCATTER TESTS FAILED");

    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
