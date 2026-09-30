#include "pathtrace.h"

#include "device_scan.cuh"
#include "geometry.cuh"
#include "interactions.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int kThreads = 256;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRayMinimum = 1.0e-4f;
#ifndef CIS565_RENDERER_BUILD_ID
#define CIS565_RENDERER_BUILD_ID \
    "cis565-project3-renderer-v1_source-manual-unspecified-build"
#endif
constexpr const char* kSourceBuild = CIS565_RENDERER_BUILD_ID;

struct RenderPath
{
    Ray ray{};
    glm::vec3 throughput{1.0f};
    glm::vec3 radiance{0.0f};
    glm::vec3 previousPosition{0.0f};
    float previousBsdfPdf = 0.0f;
    int pixelIndex = -1;
    int alive = 0;
    int previousDelta = 1;
};

struct BvhDiagnostic
{
    unsigned long long key = ~0ULL;
    int kind = -1;
    int bounce = -1;
    int pixel = -1;
    Ray ray{};
    float tMinimum = 0.0f;
    float tMaximum = 0.0f;
    int bruteResult = 0;
    int bvhResult = 0;
    Hit bruteHit{};
    Hit bvhHit{};
};

struct DeviceConfig
{
    unsigned long long seed = 1;
    unsigned long long sample = 0;
    int maxDepth = 1;
    int aa = 0;
    int nee = 0;
    int useBvh = 0;
    int lightCount = 0;
    int debugMode = 0;
    int diagnosticMode = 0;
    int pixelCount = 0;
    unsigned long long diagnosticTarget = ~0ULL;
    unsigned long long* diagnosticMinimum = nullptr;
    BvhDiagnostic* diagnosticRecord = nullptr;
};

struct LightSample
{
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 dpdu{1.0f, 0.0f, 0.0f};
    glm::vec3 dpdv{0.0f, 0.0f, 1.0f};
    glm::vec2 uv{0.0f};
    float conditionalAreaPdf = 0.0f;
    int primitiveId = -1;
    int valid = 0;
};

struct EventPair
{
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
};

struct RendererState
{
    Scene* scene = nullptr;
    Camera camera{};
    SceneView deviceView{};
    int pixelCount = 0;
    std::uint64_t completedSamples = 0;

    glm::vec3* accumulation = nullptr;
    Geom* geoms = nullptr;
    Material* materials = nullptr;
    BvhNode* nodes = nullptr;
    int* primitiveIndices = nullptr;
    TextureDescriptor* textures = nullptr;
    glm::vec3* texels = nullptr;
    int* lightIndices = nullptr;
    int lightCount = 0;

    RenderPath* paths[2] = {nullptr, nullptr};
    Hit* hits[2] = {nullptr, nullptr};
    int* flags = nullptr;
    int* offsets = nullptr;
    int* deviceCount = nullptr;
    unsigned int* deviceError = nullptr;
    unsigned long long* diagnosticMinimum = nullptr;
    BvhDiagnostic* diagnosticRecord = nullptr;
    int diagnosticMode = 0;
    unsigned long long diagnosticTarget = ~0ULL;
    std::unique_ptr<cis565::scan::ScanWorkspace> scanWorkspace;

    EventPair intersectionEvents{};
    EventPair sortEvents{};
    EventPair shadeEvents{};
    EventPair compactionEvents{};
    EventPair countEvents{};
    EventPair overallEvents{};

    std::vector<RendererBenchmarkRow> benchmarkRows;
    RendererBenchmarkSummary benchmarkSummary{};
    bool initialized = false;
};

RendererState g_state;
RendererConfig g_config;
bool g_configExplicit = false;
GuiDataContainer* g_guiData = nullptr;

void cudaCheck(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": "
            + cudaGetErrorString(result));
    }
}

void launchCheck(const char* operation)
{
    cudaCheck(cudaGetLastError(), operation);
}

template<typename T>
void allocateDevice(T** pointer, std::size_t count, const char* operation)
{
    *pointer = nullptr;
    if (count == 0)
    {
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::overflow_error(std::string(operation) + " size overflow");
    }
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)),
        operation);
}

template<typename T>
void copyVectorToDevice(T* destination, const std::vector<T>& source,
    const char* operation)
{
    if (!source.empty())
    {
        cudaCheck(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
            cudaMemcpyHostToDevice), operation);
    }
}

void createEventPair(EventPair* pair)
{
    cudaCheck(cudaEventCreate(&pair->begin), "create CUDA timing event");
    try
    {
        cudaCheck(cudaEventCreate(&pair->end), "create CUDA timing event");
    }
    catch (...)
    {
        cudaEventDestroy(pair->begin);
        pair->begin = nullptr;
        throw;
    }
}

void destroyEventPair(EventPair* pair) noexcept
{
    if (pair->begin != nullptr)
    {
        cudaEventDestroy(pair->begin);
    }
    if (pair->end != nullptr)
    {
        cudaEventDestroy(pair->end);
    }
    pair->begin = nullptr;
    pair->end = nullptr;
}

void beginTimedStage(const EventPair& pair)
{
    cudaCheck(cudaEventRecord(pair.begin), "record CUDA stage start");
}

float endTimedStage(const EventPair& pair)
{
    cudaCheck(cudaEventRecord(pair.end), "record CUDA stage end");
    cudaCheck(cudaEventSynchronize(pair.end), "wait for CUDA timed stage");
    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, pair.begin, pair.end),
        "read CUDA stage time");
    return milliseconds;
}

DeviceConfig deviceConfig()
{
    DeviceConfig result;
    result.seed = static_cast<unsigned long long>(g_config.seed);
    result.sample = static_cast<unsigned long long>(g_state.completedSamples);
    result.maxDepth = g_config.maxDepth;
    result.aa = g_config.aa ? 1 : 0;
    result.nee = g_config.nee ? 1 : 0;
    result.useBvh = g_config.useBvh ? 1 : 0;
    result.lightCount = g_state.lightCount;
    result.debugMode = static_cast<int>(g_config.debug);
    result.diagnosticMode = g_state.diagnosticMode;
    result.pixelCount = g_state.pixelCount;
    result.diagnosticTarget = g_state.diagnosticTarget;
    result.diagnosticMinimum = g_state.diagnosticMinimum;
    result.diagnosticRecord = g_state.diagnosticRecord;
    return result;
}

__host__ __device__ bool finiteScalar(float value)
{
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

__host__ __device__ bool finiteVector(const glm::vec3& value)
{
    return finiteScalar(value.x) && finiteScalar(value.y)
        && finiteScalar(value.z);
}

__device__ bool validContribution(const glm::vec3& value)
{
    return finiteVector(value) && value.x >= 0.0f && value.y >= 0.0f
        && value.z >= 0.0f;
}

__device__ void reportDeviceError(unsigned int* error, unsigned int bits)
{
    if (error != nullptr)
    {
        atomicOr(error, bits);
    }
}

__device__ glm::vec3 transformPoint(const glm::mat4& transform,
    const glm::vec3& point)
{
    return glm::vec3(transform * glm::vec4(point, 1.0f));
}

__device__ glm::vec3 transformVector(const glm::mat4& transform,
    const glm::vec3& vector)
{
    return glm::vec3(transform * glm::vec4(vector, 0.0f));
}

__device__ float linearDeterminant(const glm::mat4& transform)
{
    const glm::vec3 first(transform[0]);
    const glm::vec3 second(transform[1]);
    const glm::vec3 third(transform[2]);
    return glm::dot(first, glm::cross(second, third));
}

__device__ Ray makeCameraRay(const Camera& camera, const DeviceConfig& config,
    int pixel)
{
    const int x = pixel % camera.resolution.x;
    const int y = pixel / camera.resolution.x;
    const float jitterX = config.aa != 0
        ? renderer::randomFloat(config.seed, pixel, config.sample, -1, 0)
        : 0.5f;
    const float jitterY = config.aa != 0
        ? renderer::randomFloat(config.seed, pixel, config.sample, -1, 1)
        : 0.5f;
    const float imageX = static_cast<float>(x) + jitterX
        - 0.5f * static_cast<float>(camera.resolution.x);
    const float imageY = static_cast<float>(y) + jitterY
        - 0.5f * static_cast<float>(camera.resolution.y);
    const glm::vec3 pinholeDirection = renderer::safeNormalize(camera.view
        - camera.right * (camera.pixelLength.x * imageX)
        - camera.up * (camera.pixelLength.y * imageY), camera.view);

    Ray ray;
    ray.origin = camera.position;
    ray.direction = pinholeDirection;
    // This exact branch makes aperture zero bit-identical to a pinhole camera.
    if (!(camera.aperture > 0.0f))
    {
        return ray;
    }

    const float u0 = renderer::randomFloat(config.seed, pixel, config.sample,
        -1, 2);
    const float u1 = renderer::randomFloat(config.seed, pixel, config.sample,
        -1, 3);
    const float radius = camera.aperture * sqrtf(u0);
    const float angle = 2.0f * kPi * u1;
    const glm::vec2 lens(radius * cosf(angle), radius * sinf(angle));
    const float denominator = glm::dot(pinholeDirection, camera.view);
    const float focusParameter = camera.focusDistance
        / fmaxf(denominator, 1.0e-6f);
    const glm::vec3 focusPoint = camera.position
        + pinholeDirection * focusParameter;
    ray.origin = camera.position + lens.x * camera.right + lens.y * camera.up;
    ray.direction = renderer::safeNormalize(focusPoint - ray.origin,
        pinholeDirection);
    return ray;
}

__global__ void generatePaths(Camera camera, DeviceConfig config,
    RenderPath* paths, int pixelCount, unsigned int* deviceError)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixelCount)
    {
        return;
    }
    RenderPath path;
    path.ray = makeCameraRay(camera, config, index);
    path.throughput = glm::vec3(1.0f);
    path.radiance = glm::vec3(0.0f);
    path.previousPosition = camera.position;
    path.previousBsdfPdf = 0.0f;
    path.pixelIndex = index;
    path.alive = 1;
    path.previousDelta = 1;
    if (!finiteVector(path.ray.origin) || !finiteVector(path.ray.direction)
            || !(glm::dot(path.ray.direction, path.ray.direction) > 0.0f))
    {
        reportDeviceError(deviceError, 1U << 0U);
        path.alive = 0;
    }
    paths[index] = path;
}

__device__ bool exactHitEqual(const Hit& first, const Hit& second)
{
    return first.t == second.t
        && first.primitiveId == second.primitiveId
        && first.materialId == second.materialId
        && first.position == second.position
        && first.geometricNormal == second.geometricNormal
        && first.shadingNormal == second.shadingNormal
        && first.dpdu == second.dpdu && first.dpdv == second.dpdv
        && first.uv == second.uv && first.frontFace == second.frontFace;
}

__device__ unsigned long long diagnosticKey(const DeviceConfig& config,
    int bounce, int pixel, int kind)
{
    return (static_cast<unsigned long long>(bounce) * config.pixelCount
        + static_cast<unsigned int>(pixel)) * 2ULL
        + static_cast<unsigned int>(kind);
}

__device__ void noteBvhMismatch(const DeviceConfig& config, int kind,
    int bounce, int pixel, const Ray& ray, float tMinimum, float tMaximum,
    bool bruteResult, const Hit& bruteHit, bool bvhResult, const Hit& bvhHit)
{
    if (config.diagnosticMode == 0 || config.diagnosticMinimum == nullptr
            || config.diagnosticRecord == nullptr)
    {
        return;
    }
    const unsigned long long key = diagnosticKey(config, bounce, pixel, kind);
    if (config.diagnosticMode == 1)
    {
        atomicMin(config.diagnosticMinimum, key);
        return;
    }
    if (config.diagnosticMode == 2 && key == config.diagnosticTarget)
    {
        BvhDiagnostic result;
        result.key = key;
        result.kind = kind;
        result.bounce = bounce;
        result.pixel = pixel;
        result.ray = ray;
        result.tMinimum = tMinimum;
        result.tMaximum = tMaximum;
        result.bruteResult = bruteResult ? 1 : 0;
        result.bvhResult = bvhResult ? 1 : 0;
        result.bruteHit = bruteHit;
        result.bvhHit = bvhHit;
        *config.diagnosticRecord = result;
    }
}

__global__ void intersectPaths(SceneView scene, DeviceConfig config,
    const RenderPath* paths, Hit* hits, int count, int bounce)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    Hit hit;
    hit.t = -1.0f;
    hit.primitiveId = -1;
    hit.materialId = -1;
    if (paths[index].alive != 0 && config.diagnosticMode != 0)
    {
        Hit bruteHit;
        Hit bvhHit;
        const bool bruteFound = closestHit(scene, paths[index].ray,
            kRayMinimum, FLT_MAX, false, bruteHit);
        const bool bvhFound = closestHit(scene, paths[index].ray,
            kRayMinimum, FLT_MAX, true, bvhHit);
        if (bruteFound != bvhFound
                || (bruteFound && !exactHitEqual(bruteHit, bvhHit)))
        {
            noteBvhMismatch(config, 0, bounce, paths[index].pixelIndex,
                paths[index].ray, kRayMinimum, FLT_MAX, bruteFound, bruteHit,
                bvhFound, bvhHit);
        }
        const bool selectedFound = config.useBvh != 0 ? bvhFound : bruteFound;
        if (selectedFound)
        {
            hit = config.useBvh != 0 ? bvhHit : bruteHit;
        }
    }
    else if (paths[index].alive != 0)
    {
        Hit candidate;
        if (closestHit(scene, paths[index].ray, kRayMinimum, FLT_MAX,
                config.useBvh != 0, candidate))
        {
            hit = candidate;
        }
    }
    hits[index] = hit;
}

__device__ unsigned int materialKey(const Hit& hit, int materialCount)
{
    return hit.materialId < 0 || hit.materialId >= materialCount
        ? static_cast<unsigned int>(materialCount)
        : static_cast<unsigned int>(hit.materialId);
}

__global__ void markZeroBit(const Hit* hits, int* flags, int count, int bit,
    int materialCount)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
    {
        flags[index] = ((materialKey(hits[index], materialCount) >> bit) & 1U)
            == 0U ? 1 : 0;
    }
}

__global__ void computeFlagTotal(const int* flags, const int* offsets,
    int count, int* total)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *total = count > 0 ? offsets[count - 1] + flags[count - 1] : 0;
    }
}

__global__ void scatterRadixPairs(const RenderPath* inputPaths,
    const Hit* inputHits, RenderPath* outputPaths, Hit* outputHits,
    const int* zeroFlags, const int* zeroOffsets, const int* totalZeros,
    int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    const int destination = zeroFlags[index] != 0
        ? zeroOffsets[index]
        : *totalZeros + index - zeroOffsets[index];
    outputPaths[destination] = inputPaths[index];
    outputHits[destination] = inputHits[index];
}

__device__ void cubeUvAndTangents(const glm::vec3& localPosition,
    const glm::vec3& localNormal, glm::vec2* uv, glm::vec3* dpdu,
    glm::vec3* dpdv)
{
    if (localNormal.x > 0.0f)
    {
        *uv = glm::vec2(0.5f - localPosition.z, localPosition.y + 0.5f);
        *dpdu = glm::vec3(0.0f, 0.0f, -1.0f);
        *dpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (localNormal.x < 0.0f)
    {
        *uv = glm::vec2(localPosition.z + 0.5f, localPosition.y + 0.5f);
        *dpdu = glm::vec3(0.0f, 0.0f, 1.0f);
        *dpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (localNormal.y > 0.0f)
    {
        *uv = glm::vec2(localPosition.x + 0.5f, 0.5f - localPosition.z);
        *dpdu = glm::vec3(1.0f, 0.0f, 0.0f);
        *dpdv = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    else if (localNormal.y < 0.0f)
    {
        *uv = glm::vec2(localPosition.x + 0.5f, localPosition.z + 0.5f);
        *dpdu = glm::vec3(1.0f, 0.0f, 0.0f);
        *dpdv = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    else if (localNormal.z > 0.0f)
    {
        *uv = glm::vec2(localPosition.x + 0.5f, localPosition.y + 0.5f);
        *dpdu = glm::vec3(1.0f, 0.0f, 0.0f);
        *dpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        *uv = glm::vec2(0.5f - localPosition.x, localPosition.y + 0.5f);
        *dpdu = glm::vec3(-1.0f, 0.0f, 0.0f);
        *dpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

__device__ float primitiveAreaPdf(const Geom& geometry,
    const glm::vec3& worldPosition)
{
    if (geometry.type == TRIANGLE)
    {
        const float twiceArea = glm::length(glm::cross(
            geometry.vertices[1] - geometry.vertices[0],
            geometry.vertices[2] - geometry.vertices[0]));
        return twiceArea > 0.0f ? 2.0f / twiceArea : 0.0f;
    }
    if (geometry.type == CUBE)
    {
        const glm::vec3 xAxis(geometry.transform[0]);
        const glm::vec3 yAxis(geometry.transform[1]);
        const glm::vec3 zAxis(geometry.transform[2]);
        const float area = 2.0f * (glm::length(glm::cross(yAxis, zAxis))
            + glm::length(glm::cross(xAxis, zAxis))
            + glm::length(glm::cross(xAxis, yAxis)));
        return area > 0.0f ? 1.0f / area : 0.0f;
    }
    if (geometry.type == SPHERE)
    {
        const glm::vec3 localPosition = transformPoint(
            geometry.inverseTransform, worldPosition);
        const glm::vec3 objectNormal = renderer::safeNormalize(localPosition,
            glm::vec3(0.0f, 1.0f, 0.0f));
        const float determinant = fabsf(linearDeterminant(geometry.transform));
        const float normalScale = glm::length(transformVector(
            geometry.invTranspose, objectNormal));
        const float jacobian = 0.25f * determinant * normalScale;
        return jacobian > 0.0f ? 1.0f / (4.0f * kPi * jacobian) : 0.0f;
    }
    return 0.0f;
}

__device__ LightSample sampleLightSurface(const Geom& geometry,
    int primitiveId, float u0, float u1, float u2)
{
    LightSample sample;
    sample.primitiveId = primitiveId;
    if (geometry.type == TRIANGLE)
    {
        const glm::vec3 edge1 = geometry.vertices[1] - geometry.vertices[0];
        const glm::vec3 edge2 = geometry.vertices[2] - geometry.vertices[0];
        const glm::vec3 crossValue = glm::cross(edge1, edge2);
        const float twiceArea = glm::length(crossValue);
        if (!(twiceArea > 0.0f))
        {
            return sample;
        }
        const float root = sqrtf(u0);
        const float b0 = 1.0f - root;
        const float b1 = root * (1.0f - u1);
        const float b2 = root * u1;
        sample.position = b0 * geometry.vertices[0]
            + b1 * geometry.vertices[1] + b2 * geometry.vertices[2];
        sample.normal = crossValue / twiceArea;
        const glm::vec2 uv0 = geometry.hasUVs ? geometry.uvs[0]
            : glm::vec2(0.0f, 0.0f);
        const glm::vec2 uv1 = geometry.hasUVs ? geometry.uvs[1]
            : glm::vec2(1.0f, 0.0f);
        const glm::vec2 uv2 = geometry.hasUVs ? geometry.uvs[2]
            : glm::vec2(0.0f, 1.0f);
        sample.uv = b0 * uv0 + b1 * uv1 + b2 * uv2;
        sample.dpdu = edge1;
        sample.dpdv = edge2;
        sample.conditionalAreaPdf = 2.0f / twiceArea;
        sample.valid = 1;
        return sample;
    }
    if (geometry.type == CUBE)
    {
        const glm::vec3 xAxis(geometry.transform[0]);
        const glm::vec3 yAxis(geometry.transform[1]);
        const glm::vec3 zAxis(geometry.transform[2]);
        const float xFaceArea = glm::length(glm::cross(yAxis, zAxis));
        const float yFaceArea = glm::length(glm::cross(xAxis, zAxis));
        const float zFaceArea = glm::length(glm::cross(xAxis, yAxis));
        const float totalArea = 2.0f * (xFaceArea + yFaceArea + zFaceArea);
        if (!(totalArea > 0.0f))
        {
            return sample;
        }
        float target = u0 * totalArea;
        int axis = 0;
        float sign = -1.0f;
        const float areas[6] = {xFaceArea, xFaceArea, yFaceArea,
            yFaceArea, zFaceArea, zFaceArea};
        for (int face = 0; face < 6; ++face)
        {
            if (target < areas[face] || face == 5)
            {
                axis = face / 2;
                sign = (face & 1) != 0 ? 1.0f : -1.0f;
                break;
            }
            target -= areas[face];
        }
        glm::vec3 localPosition(0.0f);
        glm::vec3 localNormal(0.0f);
        localPosition[axis] = 0.5f * sign;
        localNormal[axis] = sign;
        const int firstAxis = (axis + 1) % 3;
        const int secondAxis = (axis + 2) % 3;
        localPosition[firstAxis] = u1 - 0.5f;
        localPosition[secondAxis] = u2 - 0.5f;
        glm::vec3 localDpdu;
        glm::vec3 localDpdv;
        cubeUvAndTangents(localPosition, localNormal, &sample.uv,
            &localDpdu, &localDpdv);
        sample.position = transformPoint(geometry.transform, localPosition);
        sample.normal = renderer::safeNormalize(transformVector(
            geometry.invTranspose, localNormal), glm::vec3(0.0f, 1.0f, 0.0f));
        sample.dpdu = transformVector(geometry.transform, localDpdu);
        sample.dpdv = transformVector(geometry.transform, localDpdv);
        sample.conditionalAreaPdf = 1.0f / totalArea;
        sample.valid = 1;
        return sample;
    }
    if (geometry.type == SPHERE)
    {
        const float y = 1.0f - 2.0f * u0;
        const float radius = sqrtf(fmaxf(0.0f, 1.0f - y * y));
        const float angle = 2.0f * kPi * u1;
        const glm::vec3 objectNormal(radius * cosf(angle), y,
            radius * sinf(angle));
        const glm::vec3 localPosition = 0.5f * objectNormal;
        sample.position = transformPoint(geometry.transform, localPosition);
        sample.normal = renderer::safeNormalize(transformVector(
            geometry.invTranspose, objectNormal), glm::vec3(0.0f, 1.0f, 0.0f));
        const float theta = atan2f(objectNormal.z, objectNormal.x);
        const float phi = asinf(fmaxf(-1.0f, fminf(1.0f, objectNormal.y)));
        sample.uv = glm::vec2(theta / (2.0f * kPi) + 0.5f,
            phi / kPi + 0.5f);
        glm::vec3 tangent;
        glm::vec3 bitangent;
        const glm::vec3 helper = fabsf(sample.normal.z) < 0.999f
            ? glm::vec3(0.0f, 0.0f, 1.0f)
            : glm::vec3(0.0f, 1.0f, 0.0f);
        tangent = renderer::safeNormalize(glm::cross(helper, sample.normal),
            glm::vec3(1.0f, 0.0f, 0.0f));
        bitangent = glm::cross(sample.normal, tangent);
        sample.dpdu = tangent;
        sample.dpdv = bitangent;
        sample.conditionalAreaPdf = primitiveAreaPdf(geometry, sample.position);
        sample.valid = sample.conditionalAreaPdf > 0.0f ? 1 : 0;
        return sample;
    }
    return sample;
}

__device__ Hit lightSampleHit(const LightSample& sample)
{
    Hit hit;
    hit.t = 0.0f;
    hit.primitiveId = sample.primitiveId;
    hit.position = sample.position;
    hit.geometricNormal = sample.normal;
    hit.shadingNormal = sample.normal;
    hit.dpdu = sample.dpdu;
    hit.dpdv = sample.dpdv;
    hit.uv = sample.uv;
    hit.frontFace = true;
    return hit;
}

__device__ void finishPath(RenderPath* path, glm::vec3* accumulation,
    unsigned int* deviceError)
{
    if (validContribution(path->radiance))
    {
        const glm::vec3 updated = accumulation[path->pixelIndex]
            + path->radiance;
        accumulation[path->pixelIndex] = updated;
        if (!validContribution(updated))
        {
            reportDeviceError(deviceError, 1U << 5U);
        }
    }
    else
    {
        reportDeviceError(deviceError, 1U << 1U);
    }
    path->alive = 0;
}

__device__ void addDirectLighting(const SceneView& scene,
    const Material* materials, int materialCount, const int* lightIndices,
    const DeviceConfig& config, int bounce, const Hit& hit,
    const renderer::SurfaceEvaluation& surface, RenderPath* path,
    unsigned int* deviceError)
{
    if (config.lightCount <= 0 || lightIndices == nullptr)
    {
        return;
    }
    const float choose = renderer::randomFloat(config.seed, path->pixelIndex,
        config.sample, bounce, 8);
    int slot = static_cast<int>(choose * config.lightCount);
    slot = slot < config.lightCount ? slot : config.lightCount - 1;
    const int primitive = lightIndices[slot];
    if (primitive < 0 || primitive >= scene.geomCount)
    {
        return;
    }
    const float u0 = renderer::randomFloat(config.seed, path->pixelIndex,
        config.sample, bounce, 9);
    const float u1 = renderer::randomFloat(config.seed, path->pixelIndex,
        config.sample, bounce, 10);
    const float u2 = renderer::randomFloat(config.seed, path->pixelIndex,
        config.sample, bounce, 11);
    const LightSample light = sampleLightSurface(scene.geoms[primitive],
        primitive, u0, u1, u2);
    const int lightMaterialId = scene.geoms[primitive].materialid;
    if (light.valid == 0 || lightMaterialId < 0
            || lightMaterialId >= materialCount)
    {
        return;
    }

    const glm::vec3 displacement = light.position - hit.position;
    const float distanceSquared = glm::dot(displacement, displacement);
    if (!(distanceSquared > 0.0f) || !finiteScalar(distanceSquared))
    {
        if (!finiteScalar(distanceSquared))
        {
            reportDeviceError(deviceError, 1U << 2U);
        }
        return;
    }
    const float distance = sqrtf(distanceSquared);
    const glm::vec3 direction = displacement / distance;
    const glm::vec3 geometric = renderer::orientedGeometricNormal(hit);
    const float geometricCosine = glm::dot(geometric, direction);
    const float surfaceCosine = glm::dot(surface.shadingNormal, direction);
    const float lightCosine = fabsf(glm::dot(light.normal, -direction));
    if (!(geometricCosine > 0.0f) || !(surfaceCosine > 0.0f)
            || !(lightCosine > 0.0f))
    {
        return;
    }

    const float lightPdf = (1.0f / config.lightCount)
        * light.conditionalAreaPdf * distanceSquared / lightCosine;
    if (!(lightPdf > 0.0f) || !finiteScalar(lightPdf))
    {
        if (!finiteScalar(lightPdf))
        {
            reportDeviceError(deviceError, 1U << 2U);
        }
        return;
    }

    Ray shadow;
    shadow.origin = renderer::offsetRayOrigin(hit.position,
        hit.geometricNormal, direction);
    const glm::vec3 shiftedDisplacement = light.position - shadow.origin;
    const float shiftedDistance = glm::length(shiftedDisplacement);
    if (!(shiftedDistance > kRayMinimum))
    {
        return;
    }
    shadow.direction = shiftedDisplacement / shiftedDistance;
    const float shadowMaximum = shiftedDistance - kRayMinimum
        * fmaxf(1.0f, shiftedDistance);
    if (!(shadowMaximum > kRayMinimum))
    {
        return;
    }
    bool occluded = false;
    if (config.diagnosticMode != 0)
    {
        const bool bruteOccluded = anyHit(scene, shadow, kRayMinimum,
            shadowMaximum, false);
        const bool bvhOccluded = anyHit(scene, shadow, kRayMinimum,
            shadowMaximum, true);
        if (bruteOccluded != bvhOccluded)
        {
            const Hit unused;
            noteBvhMismatch(config, 1, bounce, path->pixelIndex, shadow,
                kRayMinimum, shadowMaximum, bruteOccluded, unused,
                bvhOccluded, unused);
        }
        occluded = config.useBvh != 0 ? bvhOccluded : bruteOccluded;
    }
    else
    {
        occluded = anyHit(scene, shadow, kRayMinimum, shadowMaximum,
            config.useBvh != 0);
    }
    if (occluded)
    {
        return;
    }

    const Hit sampledHit = lightSampleHit(light);
    const renderer::SurfaceEvaluation lightSurface = renderer::evaluateSurface(
        scene, materials[lightMaterialId], sampledHit);
    const float bsdfPdf = renderer::cosineHemispherePdf(
        surface.shadingNormal, direction);
    const float weight = renderer::powerHeuristic(lightPdf, bsdfPdf);
    const glm::vec3 contribution = path->throughput
        * (surface.baseColor * (1.0f / kPi)) * lightSurface.emission
        * (surfaceCosine * weight / lightPdf);
    if (validContribution(contribution))
    {
        path->radiance += contribution;
    }
    else
    {
        reportDeviceError(deviceError, 1U << 1U);
    }
}

__device__ float lightPdfForHit(const SceneView& scene,
    const DeviceConfig& config, const RenderPath& path, const Hit& hit)
{
    if (config.lightCount <= 0 || hit.primitiveId < 0
            || hit.primitiveId >= scene.geomCount)
    {
        return 0.0f;
    }
    const float areaPdf = primitiveAreaPdf(scene.geoms[hit.primitiveId],
        hit.position);
    const glm::vec3 displacement = hit.position - path.previousPosition;
    const float distanceSquared = glm::dot(displacement, displacement);
    const float cosine = fabsf(glm::dot(hit.geometricNormal,
        -path.ray.direction));
    if (!(areaPdf > 0.0f) || !(distanceSquared > 0.0f) || !(cosine > 0.0f))
    {
        return 0.0f;
    }
    return (1.0f / config.lightCount) * areaPdf * distanceSquared / cosine;
}

__global__ void shadePaths(SceneView scene, const Material* materials,
    int materialCount, const int* lightIndices, DeviceConfig config,
    int bounce, RenderPath* paths, const Hit* hits, int* aliveFlags,
    glm::vec3* accumulation, int count, unsigned int* deviceError)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    RenderPath path = paths[index];
    if (path.alive == 0)
    {
        aliveFlags[index] = 0;
        return;
    }
    const Hit hit = hits[index];
    if (hit.t < 0.0f)
    {
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }
    if (!finiteScalar(hit.t) || hit.materialId < 0
            || hit.materialId >= materialCount || !finiteVector(hit.position)
            || !finiteVector(hit.geometricNormal)
            || !finiteVector(hit.shadingNormal))
    {
        reportDeviceError(deviceError, 1U << 3U);
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }

    const Material material = materials[hit.materialId];
    const renderer::SurfaceEvaluation surface = renderer::evaluateSurface(
        scene, material, hit);
    if (!validContribution(surface.baseColor)
            || !validContribution(surface.emission)
            || !finiteVector(surface.shadingNormal)
            || !(glm::dot(surface.shadingNormal, surface.shadingNormal) > 0.0f))
    {
        reportDeviceError(deviceError, 1U << 3U);
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }

    if (config.debugMode != static_cast<int>(RendererDebugMode::None))
    {
        if (config.debugMode == static_cast<int>(RendererDebugMode::Normals))
        {
            path.radiance = 0.5f * surface.shadingNormal + glm::vec3(0.5f);
        }
        else if (config.debugMode == static_cast<int>(RendererDebugMode::Uv))
        {
            path.radiance = glm::vec3(hit.uv.x - floorf(hit.uv.x),
                hit.uv.y - floorf(hit.uv.y), 0.0f);
        }
        else
        {
            const float mappedDepth = hit.t / (1.0f + hit.t);
            path.radiance = glm::vec3(mappedDepth);
        }
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }

    if (material.emittance > 0.0f)
    {
        float weight = 1.0f;
        if (config.nee != 0 && path.previousDelta == 0)
        {
            const float lightPdf = lightPdfForHit(scene, config, path, hit);
            weight = renderer::powerHeuristic(path.previousBsdfPdf, lightPdf);
        }
        const glm::vec3 emission = path.throughput * surface.emission * weight;
        if (validContribution(emission))
        {
            path.radiance += emission;
        }
        else
        {
            reportDeviceError(deviceError, 1U << 1U);
        }
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }

    const bool dielectric = material.hasRefractive > 0.0f;
    const bool mirror = !dielectric && material.hasReflective > 0.0f;
    if (bounce + 1 >= config.maxDepth)
    {
        // maxDepth counts intersections. Omitting NEE at the last one makes
        // NEE and BSDF-only modes estimate the same finite path set.
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }

    if (!dielectric && !mirror && config.nee != 0)
    {
        addDirectLighting(scene, materials, materialCount, lightIndices,
            config, bounce, hit, surface, &path, deviceError);
    }

    const glm::vec3 geometric = renderer::orientedGeometricNormal(hit);
    glm::vec3 outgoing(0.0f);
    if (dielectric)
    {
        const float etaIncident = hit.frontFace ? 1.0f
            : fmaxf(material.indexOfRefraction, 1.0e-4f);
        const float etaTransmitted = hit.frontFace
            ? fmaxf(material.indexOfRefraction, 1.0e-4f) : 1.0f;
        const float cosine = fmaxf(0.0f,
            glm::dot(-path.ray.direction, geometric));
        bool totalInternalReflection = false;
        const float fresnel = renderer::dielectricFresnel(cosine,
            etaIncident, etaTransmitted, &totalInternalReflection);
        const float choose = renderer::randomFloat(config.seed,
            path.pixelIndex, config.sample, bounce, 24);
        if (totalInternalReflection || choose < fresnel)
        {
            outgoing = renderer::safeNormalize(path.ray.direction
                - 2.0f * glm::dot(path.ray.direction, geometric) * geometric,
                geometric);
        }
        else
        {
            if (!renderer::refractDirection(path.ray.direction, geometric,
                    etaIncident, etaTransmitted, &outgoing))
            {
                outgoing = renderer::safeNormalize(path.ray.direction
                    - 2.0f * glm::dot(path.ray.direction, geometric) * geometric,
                    geometric);
            }
            else
            {
                const float eta = etaIncident / etaTransmitted;
                path.throughput *= eta * eta;
            }
        }
        path.throughput *= surface.baseColor;
        path.previousDelta = 1;
        path.previousBsdfPdf = 0.0f;
    }
    else if (mirror)
    {
        outgoing = renderer::safeNormalize(path.ray.direction
            - 2.0f * glm::dot(path.ray.direction, surface.shadingNormal)
                * surface.shadingNormal, geometric);
        if (glm::dot(outgoing, geometric) <= 0.0f)
        {
            outgoing = renderer::safeNormalize(path.ray.direction
                - 2.0f * glm::dot(path.ray.direction, geometric) * geometric,
                geometric);
        }
        path.throughput *= surface.baseColor;
        path.previousDelta = 1;
        path.previousBsdfPdf = 0.0f;
    }
    else
    {
        const float u0 = renderer::randomFloat(config.seed, path.pixelIndex,
            config.sample, bounce, 16);
        const float u1 = renderer::randomFloat(config.seed, path.pixelIndex,
            config.sample, bounce, 17);
        outgoing = renderer::cosineHemisphere(surface.shadingNormal, u0, u1);
        if (!(glm::dot(outgoing, geometric) > 0.0f))
        {
            finishPath(&path, accumulation, deviceError);
            paths[index] = path;
            aliveFlags[index] = 0;
            return;
        }
        path.throughput *= surface.baseColor;
        path.previousDelta = 0;
        path.previousBsdfPdf = renderer::cosineHemispherePdf(
            surface.shadingNormal, outgoing);
    }

    if (!validContribution(path.throughput) || !finiteVector(outgoing)
            || !(glm::dot(outgoing, outgoing) > 0.0f)
            || !finiteScalar(path.previousBsdfPdf)
            || path.previousBsdfPdf < 0.0f)
    {
        reportDeviceError(deviceError, 1U << 4U);
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }
    if (path.throughput.x == 0.0f && path.throughput.y == 0.0f
            && path.throughput.z == 0.0f)
    {
        finishPath(&path, accumulation, deviceError);
        paths[index] = path;
        aliveFlags[index] = 0;
        return;
    }
    path.previousPosition = hit.position;
    path.ray.origin = renderer::offsetRayOrigin(hit.position,
        hit.geometricNormal, outgoing);
    path.ray.direction = outgoing;
    path.alive = 1;
    paths[index] = path;
    aliveFlags[index] = 1;
}

__global__ void scatterAlivePaths(const RenderPath* input,
    RenderPath* output, const int* flags, const int* offsets, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count && flags[index] != 0)
    {
        output[offsets[index]] = input[index];
    }
}

__global__ void countAlivePaths(const RenderPath* paths, int count,
    int* activeCount)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    __shared__ int blockCounts[kThreads];
    blockCounts[threadIdx.x] = index < count && paths[index].alive != 0 ? 1 : 0;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < stride)
        {
            blockCounts[threadIdx.x] += blockCounts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(activeCount, blockCounts[0]);
    }
}

__device__ float displaySrgb(float linear)
{
    linear = fmaxf(0.0f, linear);
    const float mapped = linear / (1.0f + linear);
    return mapped <= 0.0031308f ? 12.92f * mapped
        : 1.055f * powf(mapped, 1.0f / 2.4f) - 0.055f;
}

__global__ void writeDisplay(uchar4* pbo, glm::ivec2 resolution,
    unsigned long long completedSamples, const glm::vec3* accumulation,
    int debugMode)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = resolution.x * resolution.y;
    if (index >= count || completedSamples == 0)
    {
        return;
    }
    const glm::vec3 average = accumulation[index]
        / static_cast<float>(completedSamples);
    const float red = fminf(1.0f, fmaxf(0.0f, debugMode == 0
        ? displaySrgb(average.x) : average.x));
    const float green = fminf(1.0f, fmaxf(0.0f, debugMode == 0
        ? displaySrgb(average.y) : average.y));
    const float blue = fminf(1.0f, fmaxf(0.0f, debugMode == 0
        ? displaySrgb(average.z) : average.z));
    pbo[index] = make_uchar4(static_cast<unsigned char>(red * 255.0f + 0.5f),
        static_cast<unsigned char>(green * 255.0f + 0.5f),
        static_cast<unsigned char>(blue * 255.0f + 0.5f), 255);
}

__global__ void rendererUnitValidation(int* failures)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int result = 0;

    Geom geometries[2];
    geometries[0] = Geom{};
    geometries[0].type = SPHERE;
    geometries[0].materialid = 0;
    geometries[1] = Geom{};
    geometries[1].type = TRIANGLE;
    geometries[1].materialid = 1;
    geometries[1].vertices[0] = glm::vec3(-1.0f, -1.0f, -1.0f);
    geometries[1].vertices[1] = glm::vec3(1.0f, -1.0f, -1.0f);
    geometries[1].vertices[2] = glm::vec3(0.0f, 1.0f, -1.0f);

    BvhNode node;
    node.boundsMin = glm::vec3(-1.01f, -1.01f, -1.01f);
    node.boundsMax = glm::vec3(1.01f, 1.01f, 0.51f);
    node.left = -1;
    node.right = -1;
    node.first = 0;
    node.count = 2;
    int indices[2] = {0, 1};
    SceneView scene;
    scene.geoms = geometries;
    scene.geomCount = 2;
    scene.nodes = &node;
    scene.nodeCount = 1;
    scene.primitiveIndices = indices;

    TextureDescriptor texture;
    texture.width = 2;
    texture.height = 1;
    texture.offset = 0;
    glm::vec3 textureValues[2] = {glm::vec3(0.0f), glm::vec3(1.0f)};
    SceneView textureScene;
    textureScene.textures = &texture;
    textureScene.textureCount = 1;
    textureScene.texels = textureValues;
    textureScene.texelCount = 2;
    const glm::vec3 colorMidpoint = renderer::sampleTexture(textureScene, 0,
        glm::vec2(0.5f), true);
    const glm::vec3 heightMidpoint = renderer::sampleTexture(textureScene, 0,
        glm::vec2(0.5f), false);
    if (fabsf(colorMidpoint.x - 0.5f) > 1.0e-6f
            || fabsf(heightMidpoint.x - 0.5f) > 1.0e-6f)
    {
        result |= 1 << 7;
    }

    TextureDescriptor bumpTextures[2];
    bumpTextures[0].width = 2;
    bumpTextures[0].height = 2;
    bumpTextures[0].offset = 0;
    bumpTextures[1].width = 4;
    bumpTextures[1].height = 4;
    bumpTextures[1].offset = 4;
    glm::vec3 bumpValues[20];
    for (int texel = 0; texel < 4; ++texel)
    {
        bumpValues[texel] = glm::vec3(0.375f);
    }
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            bumpValues[4 + y * 4 + x] = glm::vec3(
                static_cast<float>(x) / 3.0f);
        }
    }
    SceneView bumpScene;
    bumpScene.textures = bumpTextures;
    bumpScene.textureCount = 2;
    bumpScene.texels = bumpValues;
    bumpScene.texelCount = 20;
    Hit bumpHit;
    bumpHit.position = glm::vec3(0.0f);
    bumpHit.geometricNormal = glm::vec3(0.0f, 0.0f, 1.0f);
    bumpHit.shadingNormal = glm::vec3(0.2f, 0.3f, 0.93f);
    bumpHit.dpdu = glm::vec3(4.0f, 0.15f, 0.1f);
    bumpHit.dpdv = glm::vec3(0.1f, 0.35f, 0.02f);
    bumpHit.uv = glm::vec2(0.5f);
    bumpHit.frontFace = true;
    Material bumpMaterial;
    bumpMaterial.bumpStrength = 0.5f;
    bumpMaterial.bumpTextureId = -1;
    const renderer::SurfaceEvaluation baselineSurface =
        renderer::evaluateSurface(bumpScene, bumpMaterial, bumpHit);
    bumpMaterial.bumpTextureId = 0;
    const renderer::SurfaceEvaluation constantBump = renderer::evaluateSurface(
        bumpScene, bumpMaterial, bumpHit);
    if (constantBump.shadingNormal != baselineSurface.shadingNormal)
    {
        result |= 1 << 8;
    }
    bumpMaterial.bumpTextureId = 1;
    const renderer::SurfaceEvaluation varyingBump = renderer::evaluateSurface(
        bumpScene, bumpMaterial, bumpHit);
    const glm::vec3 bumpDifference = varyingBump.shadingNormal
        - baselineSurface.shadingNormal;
    if (!finiteVector(varyingBump.shadingNormal)
            || !(glm::dot(varyingBump.shadingNormal,
                    bumpHit.geometricNormal) > 1.0e-5f)
            || !(glm::dot(bumpDifference, bumpDifference) > 1.0e-10f))
    {
        result |= 1 << 9;
    }

    Ray ray;
    ray.origin = glm::vec3(0.0f, 0.0f, 2.0f);
    ray.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    Hit brute;
    Hit accelerated;
    const bool bruteFound = closestHit(scene, ray, 0.0f, FLT_MAX, false, brute);
    const bool bvhFound = closestHit(scene, ray, 0.0f, FLT_MAX, true, accelerated);
    if (!bruteFound || !bvhFound || brute.primitiveId != accelerated.primitiveId
            || brute.t != accelerated.t)
    {
        result |= 1 << 0;
    }
    if (anyHit(scene, ray, 0.0f, FLT_MAX, false)
            != anyHit(scene, ray, 0.0f, FLT_MAX, true))
    {
        result |= 1 << 1;
    }

    bool totalInternalReflection = false;
    const float normalFresnel = renderer::dielectricFresnel(1.0f, 1.0f,
        1.5f, &totalInternalReflection);
    if (totalInternalReflection || fabsf(normalFresnel - 0.04f) > 1.0e-6f)
    {
        result |= 1 << 2;
    }
    const float tirFresnel = renderer::dielectricFresnel(0.1f, 1.5f, 1.0f,
        &totalInternalReflection);
    if (!totalInternalReflection || tirFresnel != 1.0f)
    {
        result |= 1 << 3;
    }
    const float equalEtaFresnel = renderer::dielectricFresnel(0.0f, 1.0f,
        1.0f, &totalInternalReflection);
    const float largePdfWeight = renderer::powerHeuristic(FLT_MAX,
        0.5f * FLT_MAX);
    if (totalInternalReflection || equalEtaFresnel != 0.0f
            || !finiteScalar(largePdfWeight)
            || fabsf(largePdfWeight - 0.8f) > 2.0e-6f)
    {
        result |= 1 << 6;
    }

    const float randomA = renderer::randomFloat(7, 3, 11, 2, 5);
    const float randomB = renderer::randomFloat(7, 3, 11, 2, 5);
    const std::uint32_t randomWord = renderer::counterRandomWord(7, 3, 11,
        2, 5);
    if (!(randomA > 0.0f && randomA < 1.0f) || randomA != randomB
            || randomWord != renderer::counterRandomWord(7, 3, 11, 2, 5))
    {
        result |= 1 << 4;
    }
    if (randomWord == renderer::counterRandomWord(7, 4, 11, 2, 5)
            || randomWord == renderer::counterRandomWord(7, 3, 11, 3, 5)
            || randomWord == renderer::counterRandomWord(7, 3, 11, 2, 6))
    {
        result |= 1 << 10;
    }
    const std::uint64_t lowSeed = 0x0000000012345678ULL;
    const std::uint64_t highSeed = 0x0000000112345678ULL;
    const std::uint64_t lowSample = 0x0000000076543210ULL;
    const std::uint64_t highSample = 0x0000000176543210ULL;
    if (renderer::counterRandomWord(lowSeed, 3, lowSample, 2, 5)
                == renderer::counterRandomWord(highSeed, 3, lowSample, 2, 5)
            || renderer::counterRandomWord(lowSeed, 3, lowSample, 2, 5)
                == renderer::counterRandomWord(lowSeed, 3, highSample, 2, 5))
    {
        result |= 1 << 11;
    }
    for (int dimension = 0; dimension < 1024; ++dimension)
    {
        const float value = renderer::randomFloat(9, 17, 23, 31, dimension);
        if (!(value > 0.0f && value < 1.0f))
        {
            result |= 1 << 4;
        }
    }

    Camera camera;
    camera.resolution = glm::ivec2(4, 4);
    camera.position = glm::vec3(0.0f, 0.0f, 2.0f);
    camera.view = glm::vec3(0.0f, 0.0f, -1.0f);
    camera.right = glm::vec3(1.0f, 0.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
    camera.pixelLength = glm::vec2(0.5f);
    camera.aperture = 0.0f;
    camera.focusDistance = 2.0f;
    DeviceConfig config;
    config.aa = 0;
    const Ray generated = makeCameraRay(camera, config, 5);
    const float imageX = 1.5f - 2.0f;
    const float imageY = 1.5f - 2.0f;
    const glm::vec3 expected = renderer::safeNormalize(camera.view
        - camera.right * (camera.pixelLength.x * imageX)
        - camera.up * (camera.pixelLength.y * imageY), camera.view);
    if (generated.origin != camera.position || generated.direction != expected)
    {
        result |= 1 << 5;
    }
    *failures = result;
}

int blockCount(int count)
{
    return (count + kThreads - 1) / kThreads;
}

void requireInitialized()
{
    if (!g_state.initialized)
    {
        throw std::logic_error("path tracer is not initialized");
    }
}

void validateConfig(const RendererConfig& config)
{
    if (config.maxDepth < 1 || config.maxDepth > 4096)
    {
        throw std::invalid_argument("renderer maxDepth must be in [1, 4096]");
    }
    if (static_cast<std::uint32_t>(config.debug)
            > static_cast<std::uint32_t>(RendererDebugMode::Depth))
    {
        throw std::invalid_argument("renderer debug mode is invalid");
    }
}

void radixSortPairs(RenderPath*& paths, RenderPath*& scratchPaths,
    Hit*& hits, Hit*& scratchHits, int count, int materialCount)
{
    if (count <= 1)
    {
        return;
    }
    const int blocks = blockCount(count);
    const cis565::scan::Config scanConfig{};
    unsigned int highestKey = static_cast<unsigned int>(materialCount);
    int bitCount = 1;
    while ((highestKey >>= 1U) != 0U)
    {
        ++bitCount;
    }
    for (int bit = 0; bit < bitCount; ++bit)
    {
        markZeroBit<<<blocks, kThreads>>>(hits, g_state.flags, count, bit,
            materialCount);
        launchCheck("mark material radix bit");
        cis565::scan::scanDevice(static_cast<std::size_t>(count),
            g_state.offsets, g_state.flags, scanConfig,
            *g_state.scanWorkspace);
        computeFlagTotal<<<1, 1>>>(g_state.flags, g_state.offsets, count,
            g_state.deviceCount);
        launchCheck("compute radix zero count");
        scatterRadixPairs<<<blocks, kThreads>>>(paths, hits, scratchPaths,
            scratchHits, g_state.flags, g_state.offsets,
            g_state.deviceCount, count);
        launchCheck("scatter stable material radix pass");
        std::swap(paths, scratchPaths);
        std::swap(hits, scratchHits);
    }
}

int countAlive(const RenderPath* paths, int count, float* milliseconds)
{
    if (count <= 0)
    {
        if (milliseconds != nullptr)
        {
            *milliseconds = 0.0f;
        }
        return 0;
    }
    if (milliseconds != nullptr)
    {
        beginTimedStage(g_state.countEvents);
    }
    cudaCheck(cudaMemset(g_state.deviceCount, 0, sizeof(int)),
        "clear benchmark active counter");
    countAlivePaths<<<blockCount(count), kThreads>>>(paths, count,
        g_state.deviceCount);
    launchCheck("count active paths for benchmark");
    if (milliseconds != nullptr)
    {
        *milliseconds = endTimedStage(g_state.countEvents);
    }
    int result = 0;
    cudaCheck(cudaMemcpy(&result, g_state.deviceCount, sizeof(result),
        cudaMemcpyDeviceToHost), "download benchmark active path count");
    if (result < 0 || result > count)
    {
        throw std::runtime_error("GPU returned an invalid active path count");
    }
    return result;
}

int scanAliveCount(int count)
{
    if (count <= 0)
    {
        return 0;
    }
    const cis565::scan::Config scanConfig{};
    cis565::scan::scanDevice(static_cast<std::size_t>(count),
        g_state.offsets, g_state.flags, scanConfig, *g_state.scanWorkspace);
    computeFlagTotal<<<1, 1>>>(g_state.flags, g_state.offsets, count,
        g_state.deviceCount);
    launchCheck("compute active path count");
    int result = 0;
    cudaCheck(cudaMemcpy(&result, g_state.deviceCount, sizeof(result),
        cudaMemcpyDeviceToHost), "download active path count");
    if (result < 0 || result > count)
    {
        throw std::runtime_error("GPU returned an invalid active path count");
    }
    return result;
}

bool sameConfig(const RendererConfig& a, const RendererConfig& b)
{
    return a.seed == b.seed && a.maxDepth == b.maxDepth && a.aa == b.aa
        && a.nee == b.nee && a.useBvh == b.useBvh
        && a.compact == b.compact && a.sort == b.sort
        && a.benchmark == b.benchmark && a.debug == b.debug;
}

bool exactAccumulation(const std::vector<glm::vec3>& a,
    const std::vector<glm::vec3>& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        std::uint32_t aBits[3];
        std::uint32_t bBits[3];
        std::memcpy(aBits, &a[index], sizeof(aBits));
        std::memcpy(bBits, &b[index], sizeof(bBits));
        if (aBits[0] != bBits[0] || aBits[1] != bBits[1]
                || aBits[2] != bBits[2])
        {
            return false;
        }
    }
    return true;
}

struct AccumulationDifference
{
    std::uint64_t componentCount = 0;
    float maximumAbsoluteDelta = 0.0f;
    std::size_t firstPixel = 0;
    int firstChannel = -1;
    float firstReference = 0.0f;
    float firstCandidate = 0.0f;
};

AccumulationDifference accumulationDifference(
    const std::vector<glm::vec3>& reference,
    const std::vector<glm::vec3>& candidate)
{
    AccumulationDifference result;
    if (reference.size() != candidate.size())
    {
        result.componentCount = std::numeric_limits<std::uint64_t>::max();
        return result;
    }
    for (std::size_t pixel = 0; pixel < reference.size(); ++pixel)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            std::uint32_t referenceBits = 0;
            std::uint32_t candidateBits = 0;
            std::memcpy(&referenceBits, &reference[pixel][channel],
                sizeof(referenceBits));
            std::memcpy(&candidateBits, &candidate[pixel][channel],
                sizeof(candidateBits));
            if (referenceBits == candidateBits)
            {
                continue;
            }
            if (result.componentCount == 0)
            {
                result.firstPixel = pixel;
                result.firstChannel = channel;
                result.firstReference = reference[pixel][channel];
                result.firstCandidate = candidate[pixel][channel];
            }
            ++result.componentCount;
            const float delta = fabsf(reference[pixel][channel]
                - candidate[pixel][channel]);
            if (std::isfinite(delta))
            {
                result.maximumAbsoluteDelta = std::max(
                    result.maximumAbsoluteDelta, delta);
            }
            else
            {
                result.maximumAbsoluteDelta =
                    std::numeric_limits<float>::infinity();
            }
        }
    }
    return result;
}

void appendVec3Json(std::ostringstream& output, const glm::vec3& value)
{
    output << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

void appendVec2Json(std::ostringstream& output, const glm::vec2& value)
{
    output << '[' << value.x << ',' << value.y << ']';
}

void appendHitJson(std::ostringstream& output, bool found, const Hit& hit)
{
    output << "{\"found\":" << (found ? "true" : "false");
    if (found)
    {
        output << ",\"primitive_id\":" << hit.primitiveId
            << ",\"material_id\":" << hit.materialId
            << ",\"t\":" << hit.t << ",\"position\":";
        appendVec3Json(output, hit.position);
        output << ",\"geometric_normal\":";
        appendVec3Json(output, hit.geometricNormal);
        output << ",\"shading_normal\":";
        appendVec3Json(output, hit.shadingNormal);
        output << ",\"uv\":";
        appendVec2Json(output, hit.uv);
        output << ",\"front_face\":"
            << (hit.frontFace ? "true" : "false");
    }
    output << '}';
}

std::string bvhDifferenceJson(const AccumulationDifference& difference,
    const BvhDiagnostic* diagnostic)
{
    std::ostringstream output;
    output << std::setprecision(9)
        << "{\"different_components\":" << difference.componentCount
        << ",\"maximum_absolute_delta\":";
    if (std::isfinite(difference.maximumAbsoluteDelta))
    {
        output << difference.maximumAbsoluteDelta;
    }
    else
    {
        output << "null";
    }
    if (difference.componentCount != 0)
    {
        output << ",\"first_difference\":{\"pixel\":"
            << difference.firstPixel << ",\"channel\":"
            << difference.firstChannel << ",\"brute_sum\":"
            << difference.firstReference << ",\"bvh_sum\":"
            << difference.firstCandidate << '}';
    }
    output << ",\"ray_diagnostic\":";
    if (diagnostic == nullptr || diagnostic->key == ~0ULL)
    {
        output << "null";
    }
    else
    {
        output << "{\"kind\":\""
            << (diagnostic->kind == 0 ? "closest_hit" : "any_hit")
            << "\",\"bounce\":" << diagnostic->bounce
            << ",\"pixel\":" << diagnostic->pixel
            << ",\"origin\":";
        appendVec3Json(output, diagnostic->ray.origin);
        output << ",\"direction\":";
        appendVec3Json(output, diagnostic->ray.direction);
        output << ",\"t_min\":" << diagnostic->tMinimum
            << ",\"t_max\":" << diagnostic->tMaximum
            << ",\"brute\":";
        if (diagnostic->kind == 0)
        {
            appendHitJson(output, diagnostic->bruteResult != 0,
                diagnostic->bruteHit);
            output << ",\"bvh\":";
            appendHitJson(output, diagnostic->bvhResult != 0,
                diagnostic->bvhHit);
        }
        else
        {
            output << "{\"occluded\":"
                << (diagnostic->bruteResult != 0 ? "true" : "false")
                << "},\"bvh\":{\"occluded\":"
                << (diagnostic->bvhResult != 0 ? "true" : "false") << '}';
        }
        output << '}';
    }
    output << '}';
    return output.str();
}

std::string escapeJson(const std::string& input)
{
    std::ostringstream result;
    for (unsigned char character : input)
    {
        switch (character)
        {
            case '\"': result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\b': result << "\\b"; break;
            case '\f': result << "\\f"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (character < 0x20)
                {
                    result << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(character)
                        << std::dec;
                }
                else
                {
                    result << static_cast<char>(character);
                }
        }
    }
    return result.str();
}

} // namespace

void InitDataContainer(GuiDataContainer* guiData)
{
    g_guiData = guiData;
}

void pathtraceSetConfig(const RendererConfig& config)
{
    validateConfig(config);
    if (g_state.initialized && g_state.completedSamples != 0
            && !sameConfig(config, g_config))
    {
        throw std::logic_error(
            "cannot change renderer configuration after sampling has begun");
    }
    g_config = config;
    g_configExplicit = true;
}

RendererConfig pathtraceGetConfig()
{
    return g_config;
}

void pathtraceInit(Scene* scene)
{
    if (scene == nullptr)
    {
        throw std::invalid_argument("pathtraceInit requires a scene");
    }
    pathtraceFree();
    if (!g_configExplicit)
    {
        g_config.maxDepth = scene->state.traceDepth > 0
            ? scene->state.traceDepth : 8;
    }
    validateConfig(g_config);

    const Camera inputCamera = scene->state.camera;
    if (inputCamera.resolution.x <= 0 || inputCamera.resolution.y <= 0)
    {
        throw std::invalid_argument("camera resolution must be positive");
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(
        inputCamera.resolution.x) * static_cast<std::uint64_t>(
        inputCamera.resolution.y);
    if (pixels > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("camera resolution exceeds renderer limits");
    }
    for (const Geom& geometry : scene->geoms)
    {
        if (geometry.materialid < 0
                || geometry.materialid >= static_cast<int>(scene->materials.size()))
        {
            throw std::invalid_argument("geometry has an invalid material id");
        }
    }

    try
    {
        g_state.scene = scene;
        g_state.pixelCount = static_cast<int>(pixels);
        g_state.camera = inputCamera;
        g_state.camera.view = renderer::safeNormalize(
            inputCamera.lookAt - inputCamera.position, inputCamera.view);
        g_state.camera.right = renderer::safeNormalize(glm::cross(
            g_state.camera.view, inputCamera.up), inputCamera.right);
        g_state.camera.up = renderer::safeNormalize(glm::cross(
            g_state.camera.right, g_state.camera.view), inputCamera.up);
        if (!(g_state.camera.focusDistance > 0.0f))
        {
            g_state.camera.focusDistance = glm::length(
                inputCamera.lookAt - inputCamera.position);
        }
        if (!(g_state.camera.focusDistance > 0.0f)
                || g_state.camera.aperture < 0.0f)
        {
            throw std::invalid_argument("camera aperture/focus distance is invalid");
        }

        allocateDevice(&g_state.accumulation, pixels,
            "allocate accumulation image");
        allocateDevice(&g_state.paths[0], pixels, "allocate primary paths");
        allocateDevice(&g_state.paths[1], pixels, "allocate scratch paths");
        allocateDevice(&g_state.hits[0], pixels, "allocate primary hits");
        allocateDevice(&g_state.hits[1], pixels, "allocate scratch hits");
        allocateDevice(&g_state.flags, pixels, "allocate compaction flags");
        allocateDevice(&g_state.offsets, pixels, "allocate compaction offsets");
        allocateDevice(&g_state.deviceCount, 1, "allocate device path count");
        allocateDevice(&g_state.deviceError, 1, "allocate device error flag");
        allocateDevice(&g_state.diagnosticMinimum, 1,
            "allocate BVH diagnostic minimum");
        allocateDevice(&g_state.diagnosticRecord, 1,
            "allocate BVH diagnostic record");
        cudaCheck(cudaMemset(g_state.accumulation, 0,
            static_cast<std::size_t>(pixels) * sizeof(glm::vec3)),
            "clear accumulation image");

        allocateDevice(&g_state.geoms, scene->geoms.size(),
            "allocate geometry");
        allocateDevice(&g_state.materials, scene->materials.size(),
            "allocate materials");
        allocateDevice(&g_state.nodes, scene->bvhNodes.size(),
            "allocate BVH nodes");
        allocateDevice(&g_state.primitiveIndices,
            scene->bvhPrimitiveIndices.size(), "allocate BVH indices");
        allocateDevice(&g_state.textures, scene->textures.size(),
            "allocate texture descriptors");
        allocateDevice(&g_state.texels, scene->texels.size(),
            "allocate texture texels");
        copyVectorToDevice(g_state.geoms, scene->geoms, "upload geometry");
        copyVectorToDevice(g_state.materials, scene->materials,
            "upload materials");
        copyVectorToDevice(g_state.nodes, scene->bvhNodes,
            "upload BVH nodes");
        copyVectorToDevice(g_state.primitiveIndices,
            scene->bvhPrimitiveIndices, "upload BVH indices");
        copyVectorToDevice(g_state.textures, scene->textures,
            "upload texture descriptors");
        copyVectorToDevice(g_state.texels, scene->texels,
            "upload texture texels");

        std::vector<int> lights;
        for (std::size_t primitive = 0; primitive < scene->geoms.size(); ++primitive)
        {
            const int material = scene->geoms[primitive].materialid;
            if (scene->materials[static_cast<std::size_t>(material)].emittance > 0.0f)
            {
                lights.push_back(static_cast<int>(primitive));
            }
        }
        g_state.lightCount = static_cast<int>(lights.size());
        allocateDevice(&g_state.lightIndices, lights.size(),
            "allocate emissive primitive list");
        copyVectorToDevice(g_state.lightIndices, lights,
            "upload emissive primitive list");

        g_state.deviceView = SceneView{};
        g_state.deviceView.geoms = g_state.geoms;
        g_state.deviceView.geomCount = static_cast<int>(scene->geoms.size());
        g_state.deviceView.materials = g_state.materials;
        g_state.deviceView.materialCount = static_cast<int>(scene->materials.size());
        g_state.deviceView.nodes = g_state.nodes;
        g_state.deviceView.nodeCount = static_cast<int>(scene->bvhNodes.size());
        g_state.deviceView.primitiveIndices = g_state.primitiveIndices;
        g_state.deviceView.textures = g_state.textures;
        g_state.deviceView.textureCount = static_cast<int>(scene->textures.size());
        g_state.deviceView.texels = g_state.texels;
        g_state.deviceView.texelCount = static_cast<int>(scene->texels.size());

        g_state.scanWorkspace = std::make_unique<cis565::scan::ScanWorkspace>(
            pixels, cis565::scan::Config{});
        createEventPair(&g_state.intersectionEvents);
        createEventPair(&g_state.sortEvents);
        createEventPair(&g_state.shadeEvents);
        createEventPair(&g_state.compactionEvents);
        createEventPair(&g_state.countEvents);
        createEventPair(&g_state.overallEvents);
        g_state.initialized = true;
        cudaCheck(cudaDeviceSynchronize(), "finish path tracer initialization");
    }
    catch (...)
    {
        pathtraceFree();
        throw;
    }
}

void pathtraceFree()
{
    destroyEventPair(&g_state.intersectionEvents);
    destroyEventPair(&g_state.sortEvents);
    destroyEventPair(&g_state.shadeEvents);
    destroyEventPair(&g_state.compactionEvents);
    destroyEventPair(&g_state.countEvents);
    destroyEventPair(&g_state.overallEvents);
    g_state.scanWorkspace.reset();
    cudaFree(g_state.accumulation);
    cudaFree(g_state.geoms);
    cudaFree(g_state.materials);
    cudaFree(g_state.nodes);
    cudaFree(g_state.primitiveIndices);
    cudaFree(g_state.textures);
    cudaFree(g_state.texels);
    cudaFree(g_state.lightIndices);
    cudaFree(g_state.paths[0]);
    cudaFree(g_state.paths[1]);
    cudaFree(g_state.hits[0]);
    cudaFree(g_state.hits[1]);
    cudaFree(g_state.flags);
    cudaFree(g_state.offsets);
    cudaFree(g_state.deviceCount);
    cudaFree(g_state.deviceError);
    cudaFree(g_state.diagnosticMinimum);
    cudaFree(g_state.diagnosticRecord);
    g_state = RendererState{};
}

void pathtraceResetAccumulation()
{
    requireInitialized();
    cudaCheck(cudaMemset(g_state.accumulation, 0,
        static_cast<std::size_t>(g_state.pixelCount) * sizeof(glm::vec3)),
        "reset accumulation image");
    cudaCheck(cudaDeviceSynchronize(), "finish accumulation reset");
    g_state.completedSamples = 0;
    g_state.benchmarkRows.clear();
    g_state.benchmarkSummary = RendererBenchmarkSummary{};
}

void pathtraceRunSample()
{
    requireInitialized();
    if (g_state.completedSamples == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error("renderer sample counter overflow");
    }
    const auto hostStart = std::chrono::steady_clock::now();
    if (g_config.benchmark)
    {
        cudaCheck(cudaEventRecord(g_state.overallEvents.begin),
            "record sample GPU start");
    }

    const DeviceConfig config = deviceConfig();
    cudaCheck(cudaMemset(g_state.deviceError, 0, sizeof(unsigned int)),
        "clear renderer device error flag");
    const int pixelBlocks = blockCount(g_state.pixelCount);
    generatePaths<<<pixelBlocks, kThreads>>>(g_state.camera, config,
        g_state.paths[0], g_state.pixelCount, g_state.deviceError);
    launchCheck("generate camera paths");

    RenderPath* currentPaths = g_state.paths[0];
    RenderPath* scratchPaths = g_state.paths[1];
    int launchedPaths = g_state.pixelCount;
    int logicalActive = g_state.pixelCount;
    int tracedDepth = 0;

    const int launchDepth = g_config.debug == RendererDebugMode::None
        ? g_config.maxDepth : 1;
    for (int bounce = 0; bounce < launchDepth && launchedPaths > 0;
         ++bounce)
    {
        RendererBenchmarkRow row;
        row.sample = g_state.completedSamples;
        row.bounce = bounce;
        row.activeBefore = logicalActive;
        Hit* currentHits = g_state.hits[0];
        Hit* scratchHits = g_state.hits[1];
        const int blocks = blockCount(launchedPaths);

        if (g_config.benchmark)
        {
            beginTimedStage(g_state.intersectionEvents);
        }
        intersectPaths<<<blocks, kThreads>>>(g_state.deviceView, config,
            currentPaths, currentHits, launchedPaths, bounce);
        launchCheck("intersect path bounce");
        if (g_config.benchmark)
        {
            row.intersectionMs = endTimedStage(g_state.intersectionEvents);
        }

        if (g_config.benchmark)
        {
            beginTimedStage(g_state.sortEvents);
        }
        if (g_config.sort)
        {
            radixSortPairs(currentPaths, scratchPaths, currentHits,
                scratchHits, launchedPaths,
                static_cast<int>(g_state.scene->materials.size()));
        }
        if (g_config.benchmark)
        {
            row.sortMs = endTimedStage(g_state.sortEvents);
        }

        if (g_config.benchmark)
        {
            beginTimedStage(g_state.shadeEvents);
        }
        shadePaths<<<blocks, kThreads>>>(g_state.deviceView,
            g_state.materials,
            static_cast<int>(g_state.scene->materials.size()),
            g_state.lightIndices, config, bounce, currentPaths, currentHits,
            g_state.flags, g_state.accumulation, launchedPaths,
            g_state.deviceError);
        launchCheck("shade path bounce");
        if (g_config.benchmark)
        {
            row.shadeMs = endTimedStage(g_state.shadeEvents);
        }

        if (g_config.compact)
        {
            if (g_config.benchmark)
            {
                beginTimedStage(g_state.compactionEvents);
            }
            const int nextActive = scanAliveCount(launchedPaths);
            if (nextActive > 0)
            {
                scatterAlivePaths<<<blocks, kThreads>>>(currentPaths,
                    scratchPaths, g_state.flags, g_state.offsets,
                    launchedPaths);
                launchCheck("scatter compacted paths");
            }
            std::swap(currentPaths, scratchPaths);
            launchedPaths = nextActive;
            logicalActive = nextActive;
            if (g_config.benchmark)
            {
                row.compactionMs = endTimedStage(g_state.compactionEvents);
            }
        }
        else if (g_config.benchmark)
        {
            logicalActive = countAlive(currentPaths, launchedPaths,
                &row.countMs);
        }
        row.activeAfter = logicalActive;
        if (g_config.benchmark)
        {
            g_state.benchmarkRows.push_back(row);
        }
        tracedDepth = bounce + 1;

        // The uncompacted baseline always launches its full fixed-size array
        // for every configured depth, including under instrumentation.
        if (g_config.compact && logicalActive == 0)
        {
            break;
        }
    }

    if (g_config.benchmark)
    {
        cudaCheck(cudaEventRecord(g_state.overallEvents.end),
            "record sample GPU end");
        cudaCheck(cudaEventSynchronize(g_state.overallEvents.end),
            "wait for benchmark sample");
        float gpuMilliseconds = 0.0f;
        cudaCheck(cudaEventElapsedTime(&gpuMilliseconds,
            g_state.overallEvents.begin, g_state.overallEvents.end),
            "read sample GPU time");
        g_state.benchmarkSummary.gpuMilliseconds += gpuMilliseconds;
    }
    else
    {
        cudaCheck(cudaDeviceSynchronize(), "finish path traced sample");
    }
    unsigned int deviceError = 0;
    cudaCheck(cudaMemcpy(&deviceError, g_state.deviceError,
        sizeof(deviceError), cudaMemcpyDeviceToHost),
        "download renderer device error flag");
    if (deviceError != 0U)
    {
        std::ostringstream message;
        message << "renderer detected invalid device state (error bits 0x"
            << std::hex << deviceError << ')';
        throw std::runtime_error(message.str());
    }
    ++g_state.completedSamples;
    if (g_guiData != nullptr)
    {
        g_guiData->TracedDepth = tracedDepth;
    }
    if (g_config.benchmark)
    {
        const auto hostEnd = std::chrono::steady_clock::now();
        g_state.benchmarkSummary.hostMilliseconds +=
            std::chrono::duration<double, std::milli>(hostEnd - hostStart).count();
        ++g_state.benchmarkSummary.measuredSamples;
    }
}

void pathtrace(uchar4* pbo, int frame, int iteration)
{
    (void)frame;
    (void)iteration;
    pathtraceRunSample();
    if (pbo != nullptr)
    {
        writeDisplay<<<blockCount(g_state.pixelCount), kThreads>>>(pbo,
            g_state.camera.resolution,
            static_cast<unsigned long long>(g_state.completedSamples),
            g_state.accumulation, static_cast<int>(g_config.debug));
        launchCheck("write preview display buffer");
        cudaCheck(cudaDeviceSynchronize(), "finish preview display buffer");
    }
}

std::uint64_t pathtraceSampleCount()
{
    requireInitialized();
    return g_state.completedSamples;
}

void pathtraceDownloadAccumulation(std::vector<glm::vec3>& accumulation)
{
    requireInitialized();
    accumulation.resize(static_cast<std::size_t>(g_state.pixelCount));
    if (!accumulation.empty())
    {
        cudaCheck(cudaMemcpy(accumulation.data(), g_state.accumulation,
            accumulation.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost),
            "download accumulation image");
    }
}

void pathtraceUploadAccumulation(const std::vector<glm::vec3>& accumulation,
    std::uint64_t completedSamples)
{
    requireInitialized();
    if (accumulation.size() != static_cast<std::size_t>(g_state.pixelCount))
    {
        throw std::invalid_argument(
            "uploaded accumulation dimensions do not match the renderer");
    }
    for (const glm::vec3& value : accumulation)
    {
        if (!std::isfinite(value.x) || !std::isfinite(value.y)
                || !std::isfinite(value.z) || value.x < 0.0f
                || value.y < 0.0f || value.z < 0.0f)
        {
            throw std::invalid_argument(
                "uploaded accumulation contains invalid radiance");
        }
    }
    if (!accumulation.empty())
    {
        cudaCheck(cudaMemcpy(g_state.accumulation, accumulation.data(),
            accumulation.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice),
            "upload accumulation image");
    }
    g_state.completedSamples = completedSamples;
}

std::vector<RendererBenchmarkRow> pathtraceTakeBenchmarkRows()
{
    requireInitialized();
    std::vector<RendererBenchmarkRow> result;
    result.swap(g_state.benchmarkRows);
    return result;
}

RendererBenchmarkSummary pathtraceBenchmarkSummary()
{
    requireInitialized();
    return g_state.benchmarkSummary;
}

RendererCompatibilityInfo pathtraceCompatibilityInfo()
{
    RendererCompatibilityInfo result;
    result.sourceBuild = kSourceBuild;
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "query active CUDA device");
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, device),
        "query CUDA device properties");
    result.deviceName = properties.name;
    result.computeMajor = properties.major;
    result.computeMinor = properties.minor;
    cudaCheck(cudaRuntimeGetVersion(&result.runtimeVersion),
        "query CUDA runtime version");
    cudaCheck(cudaDriverGetVersion(&result.driverVersion),
        "query CUDA driver version");
    return result;
}

RendererValidationReport pathtraceValidate()
{
    requireInitialized();
    RendererValidationReport report;
    const RendererConfig savedConfig = g_config;
    const std::uint64_t savedSamples = g_state.completedSamples;
    const std::vector<RendererBenchmarkRow> savedRows = g_state.benchmarkRows;
    const RendererBenchmarkSummary savedSummary = g_state.benchmarkSummary;
    std::vector<glm::vec3> savedAccumulation;
    pathtraceDownloadAccumulation(savedAccumulation);

    const auto addCheck = [&](const std::string& name, bool passed,
                              const std::string& detail)
    {
        report.checks.push_back(RendererValidationCheck{name, passed, detail});
    };
    const auto restore = [&]()
    {
        g_state.diagnosticMode = 0;
        g_state.diagnosticTarget = ~0ULL;
        g_config = savedConfig;
        pathtraceUploadAccumulation(savedAccumulation, savedSamples);
        g_state.benchmarkRows = savedRows;
        g_state.benchmarkSummary = savedSummary;
    };

    try
    {
        RendererConfig referenceConfig = savedConfig;
        referenceConfig.benchmark = false;
        referenceConfig.compact = true;
        referenceConfig.sort = false;
        referenceConfig.useBvh = false;
        referenceConfig.debug = RendererDebugMode::None;

        const auto renderOne = [&](const RendererConfig& config)
        {
            g_config = config;
            pathtraceResetAccumulation();
            pathtraceRunSample();
            std::vector<glm::vec3> result;
            pathtraceDownloadAccumulation(result);
            return result;
        };

        const std::vector<glm::vec3> reference = renderOne(referenceConfig);
        RendererConfig candidateConfig = referenceConfig;
        candidateConfig.compact = false;
        const std::vector<glm::vec3> uncompacted = renderOne(candidateConfig);
        addCheck("compact_equivalence", exactAccumulation(reference, uncompacted),
            "raw float sums for compact on/off");

        candidateConfig = referenceConfig;
        candidateConfig.sort = true;
        const std::vector<glm::vec3> sorted = renderOne(candidateConfig);
        addCheck("material_sort_equivalence", exactAccumulation(reference, sorted),
            "raw float sums for material sort on/off");

        candidateConfig = referenceConfig;
        candidateConfig.useBvh = true;
        const std::vector<glm::vec3> accelerated = renderOne(candidateConfig);
        const AccumulationDifference bvhDifference = accumulationDifference(
            reference, accelerated);
        BvhDiagnostic diagnostic;
        bool haveDiagnostic = false;
        if (bvhDifference.componentCount != 0)
        {
            const unsigned long long noKey = ~0ULL;
            cudaCheck(cudaMemcpy(g_state.diagnosticMinimum, &noKey,
                sizeof(noKey), cudaMemcpyHostToDevice),
                "reset BVH diagnostic minimum");
            g_state.diagnosticMode = 1;
            g_state.diagnosticTarget = noKey;
            renderOne(candidateConfig);
            unsigned long long firstKey = noKey;
            cudaCheck(cudaMemcpy(&firstKey, g_state.diagnosticMinimum,
                sizeof(firstKey), cudaMemcpyDeviceToHost),
                "download BVH diagnostic minimum");
            if (firstKey != noKey)
            {
                diagnostic = BvhDiagnostic{};
                cudaCheck(cudaMemcpy(g_state.diagnosticRecord, &diagnostic,
                    sizeof(diagnostic), cudaMemcpyHostToDevice),
                    "reset BVH diagnostic record");
                g_state.diagnosticMode = 2;
                g_state.diagnosticTarget = firstKey;
                renderOne(candidateConfig);
                cudaCheck(cudaMemcpy(&diagnostic, g_state.diagnosticRecord,
                    sizeof(diagnostic), cudaMemcpyDeviceToHost),
                    "download BVH diagnostic record");
                haveDiagnostic = diagnostic.key == firstKey;
            }
            g_state.diagnosticMode = 0;
            g_state.diagnosticTarget = noKey;
        }
        addCheck("scene_bvh_equivalence",
            bvhDifference.componentCount == 0,
            "exact raw float sums and actual traced-ray hit decisions");
        report.checks.back().dataJson = bvhDifferenceJson(bvhDifference,
            haveDiagnostic ? &diagnostic : nullptr);

        bool finiteAndNonnegative = true;
        for (const glm::vec3& value : reference)
        {
            finiteAndNonnegative = finiteAndNonnegative
                && std::isfinite(value.x) && std::isfinite(value.y)
                && std::isfinite(value.z) && value.x >= 0.0f
                && value.y >= 0.0f && value.z >= 0.0f;
        }
        addCheck("finite_nonnegative_radiance", finiteAndNonnegative,
            "all reference accumulation components are finite and nonnegative");

        int* deviceFailures = nullptr;
        allocateDevice(&deviceFailures, 1, "allocate validation result");
        int failures = 0;
        try
        {
            rendererUnitValidation<<<1, 1>>>(deviceFailures);
            launchCheck("run renderer GPU unit validation");
            cudaCheck(cudaMemcpy(&failures, deviceFailures, sizeof(failures),
                cudaMemcpyDeviceToHost), "download GPU validation result");
            cudaFree(deviceFailures);
        }
        catch (...)
        {
            cudaFree(deviceFailures);
            throw;
        }
        addCheck("fixed_scene_bvh", (failures & 0x3) == 0,
            "hard-coded sphere/triangle brute and BVH closest/any hit");
        addCheck("dielectric_edges", (failures & 0x4c) == 0,
            "Fresnel/TIR/equal-IOR guards and overflow-safe MIS heuristic");
        addCheck("counter_rng", (failures & 0x10) == 0,
            "deterministic repeated calls and open-interval float samples");
        addCheck("counter_rng_field_identity", (failures & 0x400) == 0,
            "pixel, bounce, and dimension fields affect the ordered counter tuple");
        addCheck("counter_rng_high_halves", (failures & 0x800) == 0,
            "seed and global sample high halves affect the counter word");
        addCheck("zero_aperture_identity", (failures & 0x20) == 0,
            "thin-lens aperture zero equals the pinhole ray exactly");
        addCheck("texture_color_space", (failures & 0x80) == 0,
            "sRGB decode occurs per fetched color texel before interpolation while height remains raw");
        addCheck("bump_constant_preserves_smooth_normal",
            (failures & 0x100) == 0,
            "constant height retains the normalized interpolated shading normal exactly");
        addCheck("bump_scaled_derivatives", (failures & 0x200) == 0,
            "scaled mesh derivatives produce a finite nontrivial bump normal in the geometric hemisphere");
        restore();
    }
    catch (const std::exception& error)
    {
        try
        {
            restore();
        }
        catch (...)
        {
        }
        addCheck("validation_exception", false, error.what());
    }

    report.passed = !report.checks.empty();
    for (const RendererValidationCheck& check : report.checks)
    {
        report.passed = report.passed && check.passed;
    }
    return report;
}

std::string pathtraceValidationJson(const RendererValidationReport& report)
{
    std::ostringstream output;
    output << "{\"passed\":" << (report.passed ? "true" : "false")
        << ",\"checks\":[";
    for (std::size_t index = 0; index < report.checks.size(); ++index)
    {
        if (index != 0)
        {
            output << ',';
        }
        const RendererValidationCheck& check = report.checks[index];
        output << "{\"name\":\"" << escapeJson(check.name)
            << "\",\"passed\":" << (check.passed ? "true" : "false")
            << ",\"detail\":\"" << escapeJson(check.detail) << "\"";
        if (!check.dataJson.empty())
        {
            output << ",\"data\":" << check.dataJson;
        }
        output << '}';
    }
    output << "]}";
    return output.str();
}
