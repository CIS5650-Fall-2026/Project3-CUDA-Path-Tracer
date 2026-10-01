#include "pathtrace.h"

#include "device_scan.cuh"
#include "geometry.cuh"
#include "interactions.h"
#include "procedural_texture.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
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
constexpr int kRussianRouletteDimension = 32;
constexpr unsigned int kQueueErrorBit = 1U << 6U;
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
    double etaScale = 1.0;
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
    int rr = 0;
    int rrStart = 3;
    unsigned long long* rrCounters = nullptr;
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
    int* queueKeys = nullptr;
    int* queueIndices = nullptr;
    int* queueCounts = nullptr;
    int* queueOffsets = nullptr;
    int* queueCursors = nullptr;
    int queueKeyCount = 0;
    std::vector<int> queueHostCounts;
    std::vector<int> queueHostOffsets;
    unsigned long long* rrCounters = nullptr;
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
    EventPair queueEvents{};
    EventPair diffuseEvents{};
    EventPair mirrorEvents{};
    EventPair dielectricEvents{};
    EventPair emissionEvents{};
    EventPair terminalEvents{};
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

template<typename T>
class DeviceBuffer
{
public:
    DeviceBuffer(std::size_t count, const char* operation)
    {
        allocateDevice(&pointer_, count, operation);
    }

    ~DeviceBuffer()
    {
        cudaFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const noexcept
    {
        return pointer_;
    }

private:
    T* pointer_ = nullptr;
};

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
    result.rr = g_config.rr ? 1 : 0;
    result.rrStart = g_config.rrStart;
    result.rrCounters = g_config.benchmark ? g_state.rrCounters : nullptr;
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
    path.etaScale = 1.0;
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

__global__ void classifyMaterialQueues(const RenderPath* paths,
    const Hit* hits, int count, int materialCount, int queueCapacity,
    int* queueKeys, int* queueCounts, int* liveInputCount, int* aliveFlags,
    unsigned int* deviceError)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }

    // Classification owns the complete flag reset.  In particular, dead
    // uncompacted slots never reach a terminal queue on a later bounce.
    aliveFlags[index] = 0;
    queueKeys[index] = -1;
    if (paths[index].alive == 0)
    {
        return;
    }
    const int priorLive = atomicAdd(liveInputCount, 1);
    if (priorLive < 0 || priorLive >= queueCapacity)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
    }

    const Hit hit = hits[index];
    int key = materialCount;
    if (hit.t < 0.0f)
    {
        key = materialCount;
    }
    else if (finiteScalar(hit.t) && hit.materialId >= 0
            && hit.materialId < materialCount && finiteVector(hit.position)
            && finiteVector(hit.geometricNormal)
            && finiteVector(hit.shadingNormal))
    {
        key = hit.materialId;
    }
    else
    {
        // Match the unified shader's invalid-hit semantics while routing the
        // live input to the terminal consumer exactly once.
        reportDeviceError(deviceError, 1U << 3U);
    }

    if (key < 0 || key > materialCount)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        return;
    }
    queueKeys[index] = key;
    const int prior = atomicAdd(queueCounts + key, 1);
    if (prior < 0 || prior >= queueCapacity)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
    }
}

__global__ void validateMaterialQueuePrefix(const int* queueCounts,
    const int* queueOffsets, int queueKeyCount, int queueCapacity,
    unsigned int* deviceError)
{
    const int key = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (key >= queueKeyCount)
    {
        return;
    }
    const int count = queueCounts[key];
    const int offset = queueOffsets[key];
    const long long end = static_cast<long long>(offset) + count;
    const long long previousEnd = key == 0 ? 0LL
        : static_cast<long long>(queueOffsets[key - 1])
            + queueCounts[key - 1];
    const bool prefixMatches = key == 0 ? offset == 0
        : static_cast<long long>(offset) == previousEnd;
    if (count < 0 || count > queueCapacity || offset < 0
            || offset > queueCapacity || end < 0 || end > queueCapacity
            || !prefixMatches)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
    }
}

__global__ void scatterMaterialQueueIndices(const int* queueKeys,
    int count, int materialCount, int queueCapacity, const int* queueCounts,
    const int* queueOffsets, int* queueCursors, int* queueIndices,
    unsigned int* deviceError)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    const int key = queueKeys[index];
    if (key == -1)
    {
        return;
    }
    if (key < 0 || key > materialCount)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        return;
    }
    const int slot = atomicAdd(queueCursors + key, 1);
    const long long destination = static_cast<long long>(queueOffsets[key])
        + slot;
    if (slot < 0 || slot >= queueCounts[key] || destination < 0
            || destination >= queueCapacity)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        return;
    }
    queueIndices[static_cast<int>(destination)] = index;
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

enum class SurfacePreparation : int
{
    Finished = 0,
    Ready = 1
};

__device__ __noinline__ void finishStoredPath(int index, RenderPath* paths,
    int* aliveFlags, glm::vec3* accumulation, RenderPath* path,
    unsigned int* deviceError)
{
    finishPath(path, accumulation, deviceError);
    paths[index] = *path;
    aliveFlags[index] = 0;
}

__device__ __noinline__ SurfacePreparation prepareSurfacePath(
    const SceneView& scene, const Material* materials, int materialCount,
    const DeviceConfig& config, int index, RenderPath* paths,
    const Hit* hits, int* aliveFlags, glm::vec3* accumulation,
    unsigned int* deviceError, RenderPath* path, Hit* hit,
    Material* material, renderer::SurfaceEvaluation* surface)
{
    *path = paths[index];
    if (path->alive == 0)
    {
        aliveFlags[index] = 0;
        return SurfacePreparation::Finished;
    }
    *hit = hits[index];
    if (hit->t < 0.0f)
    {
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return SurfacePreparation::Finished;
    }
    if (!finiteScalar(hit->t) || hit->materialId < 0
            || hit->materialId >= materialCount
            || !finiteVector(hit->position)
            || !finiteVector(hit->geometricNormal)
            || !finiteVector(hit->shadingNormal))
    {
        reportDeviceError(deviceError, 1U << 3U);
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return SurfacePreparation::Finished;
    }

    *material = materials[hit->materialId];
    *surface = renderer::evaluateSurface(scene, *material, *hit);
    if (!validContribution(surface->baseColor)
            || !validContribution(surface->emission)
            || !finiteVector(surface->shadingNormal)
            || !(glm::dot(surface->shadingNormal,
                    surface->shadingNormal) > 0.0f))
    {
        reportDeviceError(deviceError, 1U << 3U);
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return SurfacePreparation::Finished;
    }

    if (config.debugMode != static_cast<int>(RendererDebugMode::None))
    {
        if (config.debugMode == static_cast<int>(RendererDebugMode::Normals))
        {
            path->radiance = 0.5f * surface->shadingNormal + glm::vec3(0.5f);
        }
        else if (config.debugMode == static_cast<int>(RendererDebugMode::Uv))
        {
            path->radiance = glm::vec3(hit->uv.x - floorf(hit->uv.x),
                hit->uv.y - floorf(hit->uv.y), 0.0f);
        }
        else
        {
            const float mappedDepth = hit->t / (1.0f + hit->t);
            path->radiance = glm::vec3(mappedDepth);
        }
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return SurfacePreparation::Finished;
    }
    return SurfacePreparation::Ready;
}

__device__ __noinline__ void shadeEmissionBody(const SceneView& scene,
    const DeviceConfig& config, const Hit& hit,
    const renderer::SurfaceEvaluation& surface, RenderPath* path,
    unsigned int* deviceError)
{
    float weight = 1.0f;
    if (config.nee != 0 && path->previousDelta == 0)
    {
        const float lightPdf = lightPdfForHit(scene, config, *path, hit);
        weight = renderer::powerHeuristic(path->previousBsdfPdf, lightPdf);
    }
    const glm::vec3 emission = path->throughput * surface.emission * weight;
    if (validContribution(emission))
    {
        path->radiance += emission;
    }
    else
    {
        reportDeviceError(deviceError, 1U << 1U);
    }
}

__device__ bool updateEtaScale(double etaIncident, double etaTransmitted,
    RenderPath* path)
{
    if (!(etaIncident > 0.0) || !(etaTransmitted > 0.0)
            || etaIncident > DBL_MAX || etaTransmitted > DBL_MAX
            || !(path->etaScale > 0.0) || path->etaScale > DBL_MAX)
    {
        return false;
    }
    const double ratio = etaTransmitted / etaIncident;
    if (!(ratio > 0.0) || ratio > DBL_MAX)
    {
        return false;
    }
    const double factor = ratio > DBL_MAX / ratio
        ? DBL_MAX : ratio * ratio;
    path->etaScale = factor > DBL_MAX / path->etaScale
        ? DBL_MAX : path->etaScale * factor;
    return path->etaScale > 0.0 && path->etaScale <= DBL_MAX;
}

__device__ __noinline__ bool shadeDielectricBody(const Material& material,
    const renderer::SurfaceEvaluation& surface, const DeviceConfig& config,
    int bounce, const Hit& hit, RenderPath* path, glm::vec3* outgoing,
    unsigned int* deviceError)
{
    const glm::vec3 geometric = renderer::orientedGeometricNormal(hit);
    const float etaIncident = hit.frontFace ? 1.0f
        : fmaxf(material.indexOfRefraction, 1.0e-4f);
    const float etaTransmitted = hit.frontFace
        ? fmaxf(material.indexOfRefraction, 1.0e-4f) : 1.0f;
    const float cosine = fmaxf(0.0f,
        glm::dot(-path->ray.direction, geometric));
    bool totalInternalReflection = false;
    const float fresnel = renderer::dielectricFresnel(cosine,
        etaIncident, etaTransmitted, &totalInternalReflection);
    const float choose = renderer::randomFloat(config.seed,
        path->pixelIndex, config.sample, bounce, 24);
    if (totalInternalReflection || choose < fresnel)
    {
        *outgoing = renderer::safeNormalize(path->ray.direction
            - 2.0f * glm::dot(path->ray.direction, geometric) * geometric,
            geometric);
    }
    else
    {
        if (!renderer::refractDirection(path->ray.direction, geometric,
                etaIncident, etaTransmitted, outgoing))
        {
            *outgoing = renderer::safeNormalize(path->ray.direction
                - 2.0f * glm::dot(path->ray.direction, geometric) * geometric,
                geometric);
        }
        else
        {
            const float eta = etaIncident / etaTransmitted;
            if (config.rr != 0)
            {
                const double etaSquared = static_cast<double>(eta)
                    * static_cast<double>(eta);
                if (!finiteScalar(eta) || !(etaSquared >= 0.0)
                        || etaSquared > FLT_MAX
                        || static_cast<double>(path->throughput.x)
                            * etaSquared > FLT_MAX
                        || static_cast<double>(path->throughput.y)
                            * etaSquared > FLT_MAX
                        || static_cast<double>(path->throughput.z)
                            * etaSquared > FLT_MAX)
                {
                    reportDeviceError(deviceError, 1U << 4U);
                    return false;
                }
            }
            path->throughput *= eta * eta;
            if (config.rr != 0 && !updateEtaScale(
                    static_cast<double>(etaIncident),
                    static_cast<double>(etaTransmitted), path))
            {
                reportDeviceError(deviceError, 1U << 4U);
                return false;
            }
        }
    }
    path->throughput *= surface.baseColor;
    path->previousDelta = 1;
    path->previousBsdfPdf = 0.0f;
    return true;
}

__device__ __noinline__ bool shadeMirrorBody(
    const renderer::SurfaceEvaluation& surface, const Hit& hit,
    RenderPath* path, glm::vec3* outgoing)
{
    const glm::vec3 geometric = renderer::orientedGeometricNormal(hit);
    *outgoing = renderer::safeNormalize(path->ray.direction
        - 2.0f * glm::dot(path->ray.direction, surface.shadingNormal)
            * surface.shadingNormal, geometric);
    if (glm::dot(*outgoing, geometric) <= 0.0f)
    {
        *outgoing = renderer::safeNormalize(path->ray.direction
            - 2.0f * glm::dot(path->ray.direction, geometric) * geometric,
            geometric);
    }
    path->throughput *= surface.baseColor;
    path->previousDelta = 1;
    path->previousBsdfPdf = 0.0f;
    return true;
}

__device__ __noinline__ bool shadeDiffuseBody(const SceneView& scene,
    const Material* materials, int materialCount, const int* lightIndices,
    const DeviceConfig& config, int bounce, const Hit& hit,
    const renderer::SurfaceEvaluation& surface, RenderPath* path,
    glm::vec3* outgoing, unsigned int* deviceError)
{
    if (config.nee != 0)
    {
        addDirectLighting(scene, materials, materialCount, lightIndices,
            config, bounce, hit, surface, path, deviceError);
    }
    const glm::vec3 geometric = renderer::orientedGeometricNormal(hit);
    const float u0 = renderer::randomFloat(config.seed, path->pixelIndex,
        config.sample, bounce, 16);
    const float u1 = renderer::randomFloat(config.seed, path->pixelIndex,
        config.sample, bounce, 17);
    *outgoing = renderer::cosineHemisphere(surface.shadingNormal, u0, u1);
    if (!(glm::dot(*outgoing, geometric) > 0.0f))
    {
        return false;
    }
    path->throughput *= surface.baseColor;
    path->previousDelta = 0;
    path->previousBsdfPdf = renderer::cosineHemispherePdf(
        surface.shadingNormal, *outgoing);
    return true;
}

__device__ double russianRouletteProbability(const RenderPath& path)
{
    const double maximum = fmax(static_cast<double>(path.throughput.x),
        fmax(static_cast<double>(path.throughput.y),
            static_cast<double>(path.throughput.z)));
    if (!(maximum >= 0.0) || maximum > DBL_MAX
            || !(path.etaScale > 0.0) || path.etaScale > DBL_MAX)
    {
        return -1.0;
    }
    const double corrected = maximum == 0.0 ? 0.0
        : path.etaScale > DBL_MAX / maximum
            ? DBL_MAX : maximum * path.etaScale;
    return fmin(0.95, fmax(0.05, corrected));
}

__device__ __noinline__ bool applyRussianRoulette(
    const DeviceConfig& config, int bounce, RenderPath* path,
    unsigned int* deviceError)
{
    if (config.rr == 0 || bounce + 1 < config.rrStart)
    {
        return true;
    }
    const double probability = russianRouletteProbability(*path);
    if (!(probability > 0.0) || probability > 1.0)
    {
        reportDeviceError(deviceError, 1U << 4U);
        return false;
    }
    const float choose = renderer::randomFloat(config.seed,
        path->pixelIndex, config.sample, bounce,
        kRussianRouletteDimension);
    if (config.rrCounters != nullptr)
    {
        atomicAdd(config.rrCounters, 1ULL);
    }
    if (static_cast<double>(choose) >= probability)
    {
        if (config.rrCounters != nullptr)
        {
            atomicAdd(config.rrCounters + 1, 1ULL);
        }
        return false;
    }

    if (static_cast<double>(path->throughput.x) / probability > FLT_MAX
            || static_cast<double>(path->throughput.y) / probability > FLT_MAX
            || static_cast<double>(path->throughput.z) / probability > FLT_MAX)
    {
        reportDeviceError(deviceError, 1U << 4U);
        return false;
    }
    path->throughput = glm::vec3(
        static_cast<float>(static_cast<double>(path->throughput.x)
            / probability),
        static_cast<float>(static_cast<double>(path->throughput.y)
            / probability),
        static_cast<float>(static_cast<double>(path->throughput.z)
            / probability));
    return true;
}

__device__ __noinline__ void completeScatteredPath(int index, int bounce,
    const DeviceConfig& config, const Hit& hit, const glm::vec3& outgoing,
    RenderPath* paths, int* aliveFlags, glm::vec3* accumulation,
    RenderPath* path, unsigned int* deviceError)
{
    if (!validContribution(path->throughput) || !finiteVector(outgoing)
            || !(glm::dot(outgoing, outgoing) > 0.0f)
            || !finiteScalar(path->previousBsdfPdf)
            || path->previousBsdfPdf < 0.0f)
    {
        reportDeviceError(deviceError, 1U << 4U);
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return;
    }
    if (path->throughput.x == 0.0f && path->throughput.y == 0.0f
            && path->throughput.z == 0.0f)
    {
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return;
    }
    if (!applyRussianRoulette(config, bounce, path, deviceError))
    {
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return;
    }
    if (!validContribution(path->throughput))
    {
        reportDeviceError(deviceError, 1U << 4U);
        finishStoredPath(index, paths, aliveFlags, accumulation, path,
            deviceError);
        return;
    }
    path->previousPosition = hit.position;
    path->ray.origin = renderer::offsetRayOrigin(hit.position,
        hit.geometricNormal, outgoing);
    path->ray.direction = outgoing;
    path->alive = 1;
    paths[index] = *path;
    aliveFlags[index] = 1;
}

__device__ bool finalDepthTerminates(int index, int bounce,
    const DeviceConfig& config, RenderPath* paths, int* aliveFlags,
    glm::vec3* accumulation, RenderPath* path, unsigned int* deviceError)
{
    if (bounce + 1 < config.maxDepth)
    {
        return false;
    }
    // maxDepth counts intersections. Omitting NEE at the last one makes NEE
    // and BSDF-only modes estimate the same finite path set.
    finishStoredPath(index, paths, aliveFlags, accumulation, path,
        deviceError);
    return true;
}

__device__ bool validQueueIndex(const int* queueIndices, int queueOffset,
    int queueCount, int lane, int pathCount, int* index,
    unsigned int* deviceError)
{
    if (lane >= queueCount)
    {
        return false;
    }
    const long long queuePosition = static_cast<long long>(queueOffset) + lane;
    if (queueOffset < 0 || queueCount < 0 || queuePosition < 0
            || queuePosition >= pathCount)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        return false;
    }
    *index = queueIndices[static_cast<int>(queuePosition)];
    if (*index < 0 || *index >= pathCount)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        return false;
    }
    return true;
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
    RenderPath path;
    Hit hit;
    Material material;
    renderer::SurfaceEvaluation surface;
    if (prepareSurfacePath(scene, materials, materialCount, config, index,
            paths, hits, aliveFlags, accumulation, deviceError, &path, &hit,
            &material, &surface) != SurfacePreparation::Ready)
    {
        return;
    }
    if (material.emittance > 0.0f)
    {
        shadeEmissionBody(scene, config, hit, surface, &path, deviceError);
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    if (finalDepthTerminates(index, bounce, config, paths, aliveFlags,
            accumulation, &path, deviceError))
    {
        return;
    }

    glm::vec3 outgoing(0.0f);
    bool scattered = false;
    if (material.hasRefractive > 0.0f)
    {
        scattered = shadeDielectricBody(material, surface, config, bounce,
            hit, &path, &outgoing, deviceError);
    }
    else if (material.hasReflective > 0.0f)
    {
        scattered = shadeMirrorBody(surface, hit, &path, &outgoing);
    }
    else
    {
        scattered = shadeDiffuseBody(scene, materials, materialCount,
            lightIndices, config, bounce, hit, surface, &path, &outgoing,
            deviceError);
    }
    if (!scattered)
    {
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    completeScatteredPath(index, bounce, config, hit, outgoing, paths,
        aliveFlags, accumulation, &path, deviceError);
}

__global__ void shadeDiffuseQueue(SceneView scene, const Material* materials,
    int materialCount, const int* lightIndices, DeviceConfig config,
    int bounce, RenderPath* paths, const Hit* hits, int* aliveFlags,
    glm::vec3* accumulation, const int* queueIndices, int queueOffset,
    int queueCount, int pathCount, unsigned int* deviceError)
{
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int index = -1;
    if (!validQueueIndex(queueIndices, queueOffset, queueCount, lane,
            pathCount, &index, deviceError))
    {
        return;
    }
    RenderPath path;
    Hit hit;
    Material material;
    renderer::SurfaceEvaluation surface;
    if (prepareSurfacePath(scene, materials, materialCount, config, index,
            paths, hits, aliveFlags, accumulation, deviceError, &path, &hit,
            &material, &surface) != SurfacePreparation::Ready)
    {
        return;
    }
    if (material.emittance > 0.0f || material.hasRefractive > 0.0f
            || material.hasReflective > 0.0f)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    if (finalDepthTerminates(index, bounce, config, paths, aliveFlags,
            accumulation, &path, deviceError))
    {
        return;
    }
    glm::vec3 outgoing(0.0f);
    if (!shadeDiffuseBody(scene, materials, materialCount, lightIndices,
            config, bounce, hit, surface, &path, &outgoing, deviceError))
    {
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    completeScatteredPath(index, bounce, config, hit, outgoing, paths,
        aliveFlags, accumulation, &path, deviceError);
}

__global__ void shadeMirrorQueue(SceneView scene, const Material* materials,
    int materialCount, DeviceConfig config, int bounce, RenderPath* paths,
    const Hit* hits, int* aliveFlags, glm::vec3* accumulation,
    const int* queueIndices, int queueOffset, int queueCount, int pathCount,
    unsigned int* deviceError)
{
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int index = -1;
    if (!validQueueIndex(queueIndices, queueOffset, queueCount, lane,
            pathCount, &index, deviceError))
    {
        return;
    }
    RenderPath path;
    Hit hit;
    Material material;
    renderer::SurfaceEvaluation surface;
    if (prepareSurfacePath(scene, materials, materialCount, config, index,
            paths, hits, aliveFlags, accumulation, deviceError, &path, &hit,
            &material, &surface) != SurfacePreparation::Ready)
    {
        return;
    }
    if (material.emittance > 0.0f || material.hasRefractive > 0.0f
            || !(material.hasReflective > 0.0f))
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    if (finalDepthTerminates(index, bounce, config, paths, aliveFlags,
            accumulation, &path, deviceError))
    {
        return;
    }
    glm::vec3 outgoing(0.0f);
    shadeMirrorBody(surface, hit, &path, &outgoing);
    completeScatteredPath(index, bounce, config, hit, outgoing, paths,
        aliveFlags, accumulation, &path, deviceError);
}

__global__ void shadeDielectricQueue(SceneView scene,
    const Material* materials, int materialCount, DeviceConfig config,
    int bounce, RenderPath* paths, const Hit* hits, int* aliveFlags,
    glm::vec3* accumulation, const int* queueIndices, int queueOffset,
    int queueCount, int pathCount, unsigned int* deviceError)
{
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int index = -1;
    if (!validQueueIndex(queueIndices, queueOffset, queueCount, lane,
            pathCount, &index, deviceError))
    {
        return;
    }
    RenderPath path;
    Hit hit;
    Material material;
    renderer::SurfaceEvaluation surface;
    if (prepareSurfacePath(scene, materials, materialCount, config, index,
            paths, hits, aliveFlags, accumulation, deviceError, &path, &hit,
            &material, &surface) != SurfacePreparation::Ready)
    {
        return;
    }
    if (material.emittance > 0.0f || !(material.hasRefractive > 0.0f))
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    if (finalDepthTerminates(index, bounce, config, paths, aliveFlags,
            accumulation, &path, deviceError))
    {
        return;
    }
    glm::vec3 outgoing(0.0f);
    if (!shadeDielectricBody(material, surface, config, bounce, hit, &path,
            &outgoing, deviceError))
    {
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    completeScatteredPath(index, bounce, config, hit, outgoing, paths,
        aliveFlags, accumulation, &path, deviceError);
}

__global__ void shadeEmissionQueue(SceneView scene, const Material* materials,
    int materialCount, DeviceConfig config, RenderPath* paths,
    const Hit* hits, int* aliveFlags, glm::vec3* accumulation,
    const int* queueIndices, int queueOffset, int queueCount, int pathCount,
    unsigned int* deviceError)
{
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int index = -1;
    if (!validQueueIndex(queueIndices, queueOffset, queueCount, lane,
            pathCount, &index, deviceError))
    {
        return;
    }
    RenderPath path;
    Hit hit;
    Material material;
    renderer::SurfaceEvaluation surface;
    if (prepareSurfacePath(scene, materials, materialCount, config, index,
            paths, hits, aliveFlags, accumulation, deviceError, &path, &hit,
            &material, &surface) != SurfacePreparation::Ready)
    {
        return;
    }
    if (!(material.emittance > 0.0f))
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        finishStoredPath(index, paths, aliveFlags, accumulation, &path,
            deviceError);
        return;
    }
    shadeEmissionBody(scene, config, hit, surface, &path, deviceError);
    finishStoredPath(index, paths, aliveFlags, accumulation, &path,
        deviceError);
}

__global__ void shadeTerminalQueue(int materialCount, RenderPath* paths,
    const Hit* hits, int* aliveFlags, glm::vec3* accumulation,
    const int* queueIndices, int queueOffset, int queueCount, int pathCount,
    unsigned int* deviceError)
{
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int index = -1;
    if (!validQueueIndex(queueIndices, queueOffset, queueCount, lane,
            pathCount, &index, deviceError))
    {
        return;
    }
    RenderPath path = paths[index];
    if (path.alive == 0)
    {
        reportDeviceError(deviceError, kQueueErrorBit);
        aliveFlags[index] = 0;
        return;
    }
    const Hit hit = hits[index];
    if (!(hit.t < 0.0f))
    {
        if (finiteScalar(hit.t) && hit.materialId >= 0
                && hit.materialId < materialCount
                && finiteVector(hit.position)
                && finiteVector(hit.geometricNormal)
                && finiteVector(hit.shadingNormal))
        {
            reportDeviceError(deviceError, kQueueErrorBit);
        }
        else
        {
            reportDeviceError(deviceError, 1U << 3U);
        }
    }
    finishStoredPath(index, paths, aliveFlags, accumulation, &path,
        deviceError);
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

__device__ bool validationVectorNear(const glm::vec3& actual,
    const glm::vec3& expected, float tolerance)
{
    return fabsf(actual.x - expected.x) <= tolerance
        && fabsf(actual.y - expected.y) <= tolerance
        && fabsf(actual.z - expected.z) <= tolerance;
}

__device__ bool validationUnitColor(const glm::vec3& value)
{
    return finiteVector(value) && value.x >= 0.0f && value.x <= 1.0f
        && value.y >= 0.0f && value.y <= 1.0f
        && value.z >= 0.0f && value.z <= 1.0f;
}

__global__ void proceduralTextureUnitValidation(int* failures)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int result = 0;
    constexpr float tolerance = 2.0e-6f;

    const glm::vec3 marbleOrigin = renderer::evaluateProceduralColor(2,
        glm::vec2(0.0f));
    if (!validationVectorNear(marbleOrigin,
            glm::vec3(0.7715625f, 0.6825f, 0.481875f), tolerance))
    {
        result |= 1 << 0;
    }

    const glm::vec3 cellsOrigin = renderer::evaluateProceduralColor(3,
        glm::vec2(0.0f));
    const glm::vec3 cellsCenter = renderer::evaluateProceduralColor(3,
        glm::vec2(0.0625f, 0.0625f));
    if (!validationVectorNear(cellsOrigin,
            glm::vec3(0.025f, 0.035f, 0.05f), tolerance)
            || !validationVectorNear(cellsCenter,
                glm::vec3(0.86f, 0.57f, 0.14f), tolerance))
    {
        result |= 1 << 1;
    }

    bool distinct = false;
    for (int y = -32; y <= 32; ++y)
    {
        for (int x = -32; x <= 32; ++x)
        {
            const glm::vec2 variedUv(
                static_cast<float>(17 * x + 5 * y) / 37.0f,
                static_cast<float>(19 * y - 3 * x) / 41.0f);
            const glm::vec3 marble = renderer::evaluateProceduralColor(2,
                variedUv);
            const glm::vec3 cells = renderer::evaluateProceduralColor(3,
                variedUv);
            if (!validationUnitColor(marble) || !validationUnitColor(cells))
            {
                result |= 1 << 2;
            }
            const glm::vec3 difference = marble - cells;
            distinct = distinct || glm::dot(difference, difference) > 1.0e-8f;
        }
    }
    if (!distinct)
    {
        result |= 1 << 2;
    }

    for (int j = -16; j <= 16; ++j)
    {
        for (int i = -16; i <= 16; ++i)
        {
            const glm::vec2 dyadicUv(static_cast<float>(i) / 32.0f,
                static_cast<float>(j) / 32.0f);
            const glm::vec2 shiftedUv = dyadicUv + glm::vec2(-2.0f, 3.0f);
            if (renderer::evaluateProceduralColor(2, dyadicUv)
                        != renderer::evaluateProceduralColor(2, shiftedUv)
                    || renderer::evaluateProceduralColor(3, dyadicUv)
                        != renderer::evaluateProceduralColor(3, shiftedUv))
            {
                result |= 1 << 3;
            }
        }
    }

    SceneView emptyScene;
    Hit hit;
    hit.t = 1.0f;
    hit.materialId = 0;
    hit.position = glm::vec3(0.0f);
    hit.geometricNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    hit.shadingNormal = hit.geometricNormal;
    hit.dpdu = glm::vec3(1.0f, 0.0f, 0.0f);
    hit.dpdv = glm::vec3(0.0f, 0.0f, 1.0f);
    hit.uv = glm::vec2(0.75f, 0.0f);
    hit.frontFace = true;
    Material material;
    material.color = glm::vec3(0.2f, 0.4f, 0.8f);
    material.checkerScale = 2.0f;
    bool surfaceMatches = true;
    for (int proceduralId = 2; proceduralId <= 3; ++proceduralId)
    {
        material.procedural = proceduralId;
        const renderer::SurfaceEvaluation surface = renderer::evaluateSurface(
            emptyScene, material, hit);
        const glm::vec3 expected = material.color
            * renderer::evaluateProceduralColor(proceduralId, hit.uv);
        surfaceMatches = surfaceMatches
            && validationVectorNear(surface.baseColor, expected, tolerance);
    }
    if (!surfaceMatches)
    {
        // At this UV and scale the checker branch would multiply by 0.18, so
        // this also proves IDs 2/3 are consumed only by their own evaluator.
        result |= 1 << 4;
    }
    *failures = result;
}

__global__ void russianRouletteUnitValidation(int* failures,
    unsigned long long* diagnostics)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int result = 0;

    RenderPath low;
    low.throughput = glm::vec3(1.0e-4f, 2.0e-4f, 3.0e-4f);
    low.etaScale = 1.0;
    RenderPath high;
    high.throughput = glm::vec3(20.0f, 1.0f, 0.5f);
    high.etaScale = 1.0;
    if (russianRouletteProbability(low) != 0.05
            || russianRouletteProbability(high) != 0.95)
    {
        result |= 1 << 0;
    }

    RenderPath compensated;
    compensated.throughput = glm::vec3(0.8f);
    compensated.throughput *= 4.0f;
    compensated.etaScale = 1.0;
    RenderPath wideCompensation;
    wideCompensation.throughput = glm::vec3(1.0f);
    wideCompensation.etaScale = DBL_MAX * 0.5;
    if (!updateEtaScale(2.0, 1.0, &compensated)
            || fabs(russianRouletteProbability(compensated) - 0.8)
                > 1.0e-6
            || !updateEtaScale(1.0, static_cast<double>(FLT_MAX),
                &wideCompensation)
            || wideCompensation.etaScale != DBL_MAX
            || russianRouletteProbability(wideCompensation) != 0.95)
    {
        result |= 1 << 1;
    }

    diagnostics[0] = 0ULL;
    diagnostics[1] = 0ULL;
    DeviceConfig config;
    config.rr = 1;
    config.rrStart = 1;
    config.sample = 29;
    config.rrCounters = diagnostics;
    RenderPath templatePath;
    templatePath.throughput = glm::vec3(0.5f, 0.25f, 0.125f);
    templatePath.radiance = glm::vec3(0.125f, 0.25f, 0.5f);
    templatePath.pixelIndex = 17;
    templatePath.previousBsdfPdf = 0.375f;
    templatePath.etaScale = 1.0;

    unsigned long long surviveSeed = 0ULL;
    unsigned long long killSeed = 0ULL;
    for (unsigned long long seed = 1ULL;
            seed < 100000ULL && (surviveSeed == 0ULL || killSeed == 0ULL);
            ++seed)
    {
        const float choose = renderer::randomFloat(seed,
            templatePath.pixelIndex, config.sample, 0,
            kRussianRouletteDimension);
        if (choose < 0.5f && surviveSeed == 0ULL)
        {
            surviveSeed = seed;
        }
        if (choose >= 0.5f && killSeed == 0ULL)
        {
            killSeed = seed;
        }
    }
    RenderPath survived = templatePath;
    config.seed = surviveSeed;
    const bool didSurvive = surviveSeed != 0ULL
        && applyRussianRoulette(config, 0, &survived, nullptr);
    RenderPath killed = templatePath;
    config.seed = killSeed;
    const bool didKill = killSeed != 0ULL
        && !applyRussianRoulette(config, 0, &killed, nullptr);
    if (!didSurvive || !didKill || diagnostics[0] != 2ULL
            || diagnostics[1] != 1ULL)
    {
        result |= 1 << 2;
    }
    const glm::vec3 expectedRadiance = templatePath.radiance;
    const glm::vec3 expectedMean = 0.5f * survived.throughput;
    RenderPath fractional = templatePath;
    fractional.throughput = glm::vec3(0.3f, 0.1f, 0.05f);
    fractional.etaScale = 1.1;
    const glm::vec3 fractionalOriginal = fractional.throughput;
    const double fractionalProbability = russianRouletteProbability(fractional);
    config.rrCounters = nullptr;
    unsigned long long fractionalSeed = 0ULL;
    for (unsigned long long seed = 1ULL; seed < 100000ULL; ++seed)
    {
        if (static_cast<double>(renderer::randomFloat(seed,
                fractional.pixelIndex, config.sample, 0,
                kRussianRouletteDimension)) < fractionalProbability)
        {
            fractionalSeed = seed;
            break;
        }
    }
    config.seed = fractionalSeed;
    const bool fractionalSurvived = fractionalSeed != 0ULL
        && applyRussianRoulette(config, 0, &fractional, nullptr);
    const double fractionalError = fmax(fabs(
        fractionalProbability * static_cast<double>(fractional.throughput.x)
            - static_cast<double>(fractionalOriginal.x)), fmax(fabs(
        fractionalProbability * static_cast<double>(fractional.throughput.y)
            - static_cast<double>(fractionalOriginal.y)), fabs(
        fractionalProbability * static_cast<double>(fractional.throughput.z)
            - static_cast<double>(fractionalOriginal.z))));
    if (survived.radiance != expectedRadiance
            || killed.radiance != expectedRadiance
            || survived.previousBsdfPdf != templatePath.previousBsdfPdf
            || killed.previousBsdfPdf != templatePath.previousBsdfPdf
            || glm::length(expectedMean - templatePath.throughput) > 1.0e-6f
            || !fractionalSurvived
            || fractional.radiance != expectedRadiance
            || fractionalError > 1.0e-6)
    {
        result |= 1 << 3;
    }

    const unsigned long long testedBefore = diagnostics[0];
    RenderPath ineligible = templatePath;
    config.rrStart = 2;
    config.rrCounters = diagnostics;
    if (!applyRussianRoulette(config, 0, &ineligible, nullptr)
            || ineligible.throughput != templatePath.throughput
            || ineligible.radiance != templatePath.radiance
            || diagnostics[0] != testedBefore)
    {
        result |= 1 << 4;
    }

    diagnostics[0] = 0ULL;
    diagnostics[1] = 0ULL;
    config.rr = 1;
    config.rrStart = 1;
    config.rrCounters = diagnostics;
    config.debugMode = static_cast<int>(RendererDebugMode::None);
    config.maxDepth = 1;
    SceneView exclusionScene;
    Material exclusionMaterials[1];
    RenderPath exclusionPaths[1];
    Hit exclusionHits[1];
    int exclusionFlags[1] = {1};
    glm::vec3 exclusionAccumulation[1] = {glm::vec3(0.0f)};

    exclusionPaths[0] = RenderPath{};
    exclusionPaths[0].pixelIndex = 0;
    exclusionPaths[0].alive = 1;
    finishStoredPath(0, exclusionPaths, exclusionFlags,
        exclusionAccumulation, &exclusionPaths[0], nullptr);
    bool exclusionsPassed = exclusionPaths[0].alive == 0
        && exclusionFlags[0] == 0 && diagnostics[0] == 0ULL
        && diagnostics[1] == 0ULL;

    exclusionPaths[0] = RenderPath{};
    exclusionPaths[0].pixelIndex = 0;
    exclusionPaths[0].alive = 1;
    exclusionHits[0] = Hit{};
    exclusionHits[0].t = 1.0f;
    exclusionHits[0].materialId = 0;
    exclusionHits[0].position = glm::vec3(0.0f);
    exclusionHits[0].geometricNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    exclusionHits[0].shadingNormal = exclusionHits[0].geometricNormal;
    exclusionHits[0].frontFace = true;
    exclusionFlags[0] = 1;
    exclusionAccumulation[0] = glm::vec3(0.0f);
    config.debugMode = static_cast<int>(RendererDebugMode::Normals);
    RenderPath debugPath;
    Hit debugHit;
    Material debugMaterial;
    renderer::SurfaceEvaluation debugSurface;
    exclusionsPassed = exclusionsPassed
        && prepareSurfacePath(exclusionScene, exclusionMaterials, 1, config,
            0, exclusionPaths, exclusionHits, exclusionFlags,
            exclusionAccumulation, nullptr, &debugPath, &debugHit,
            &debugMaterial, &debugSurface) == SurfacePreparation::Finished
        && exclusionPaths[0].alive == 0 && exclusionFlags[0] == 0
        && diagnostics[0] == 0ULL && diagnostics[1] == 0ULL;

    exclusionPaths[0] = RenderPath{};
    exclusionPaths[0].pixelIndex = 0;
    exclusionPaths[0].alive = 1;
    exclusionFlags[0] = 1;
    exclusionAccumulation[0] = glm::vec3(0.0f);
    config.debugMode = static_cast<int>(RendererDebugMode::None);
    RenderPath finalPath;
    Hit finalHit;
    Material finalMaterial;
    renderer::SurfaceEvaluation finalSurface;
    const bool finalPrepared = prepareSurfacePath(exclusionScene,
        exclusionMaterials, 1, config, 0, exclusionPaths, exclusionHits,
        exclusionFlags, exclusionAccumulation, nullptr, &finalPath,
        &finalHit, &finalMaterial, &finalSurface)
        == SurfacePreparation::Ready;
    const bool finalTerminated = finalPrepared && finalDepthTerminates(0, 0,
        config, exclusionPaths, exclusionFlags, exclusionAccumulation,
        &finalPath, nullptr);
    exclusionsPassed = exclusionsPassed && finalTerminated
        && exclusionPaths[0].alive == 0 && exclusionFlags[0] == 0
        && diagnostics[0] == 0ULL && diagnostics[1] == 0ULL;
    if (!exclusionsPassed)
    {
        result |= 1 << 5;
    }
    *failures = result;
}

int blockCount(int count)
{
    return count / kThreads + (count % kThreads != 0 ? 1 : 0);
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
    if (config.rrStart < 1 || config.rrStart > 4096)
    {
        throw std::invalid_argument("renderer rrStart must be in [1, 4096]");
    }
    if (config.wavefront && config.sort)
    {
        throw std::invalid_argument("--sort on requires --wavefront off");
    }
    if (static_cast<std::uint32_t>(config.debug)
            > static_cast<std::uint32_t>(RendererDebugMode::Depth))
    {
        throw std::invalid_argument("renderer debug mode is invalid");
    }
}

struct MaterialQueueBuffers
{
    int* keys = nullptr;
    int* indices = nullptr;
    int* counts = nullptr;
    int* offsets = nullptr;
    int* cursors = nullptr;
    int* liveCount = nullptr;
    int capacity = 0;
};

void enqueueMaterialQueues(const RenderPath* paths, const Hit* hits,
    int count, int materialCount, int* aliveFlags,
    const MaterialQueueBuffers& buffers, unsigned int* deviceError,
    cis565::scan::ScanWorkspace& scanWorkspace)
{
    if (count < 0 || materialCount < 0 || buffers.capacity < count
            || materialCount == std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("invalid material queue dimensions");
    }
    const int keyCount = materialCount + 1;
    if (buffers.counts == nullptr || buffers.offsets == nullptr
            || buffers.cursors == nullptr || buffers.liveCount == nullptr
            || aliveFlags == nullptr
            || deviceError == nullptr
            || (count > 0 && (paths == nullptr || hits == nullptr
                || buffers.keys == nullptr || buffers.indices == nullptr)))
    {
        throw std::invalid_argument("material queue buffers are incomplete");
    }
    if (static_cast<std::size_t>(keyCount) > scanWorkspace.capacity())
    {
        throw std::invalid_argument("material queue scan workspace is too small");
    }

    const std::size_t queueBytes = static_cast<std::size_t>(keyCount)
        * sizeof(int);
    cudaCheck(cudaMemset(buffers.counts, 0, queueBytes),
        "clear material queue counts");
    cudaCheck(cudaMemset(buffers.cursors, 0, queueBytes),
        "clear material queue cursors");
    cudaCheck(cudaMemset(buffers.liveCount, 0, sizeof(int)),
        "clear material queue live-input counter");
    if (count > 0)
    {
        classifyMaterialQueues<<<blockCount(count), kThreads>>>(paths, hits,
            count, materialCount, buffers.capacity, buffers.keys,
            buffers.counts, buffers.liveCount, aliveFlags, deviceError);
        launchCheck("classify material queues");
    }
    const cis565::scan::Config scanConfig{};
    cis565::scan::scanDevice(static_cast<std::size_t>(keyCount),
        buffers.offsets, buffers.counts, scanConfig, scanWorkspace);
    validateMaterialQueuePrefix<<<blockCount(keyCount), kThreads>>>(
        buffers.counts, buffers.offsets, keyCount, buffers.capacity,
        deviceError);
    launchCheck("validate material queue prefix");
    if (count > 0)
    {
        scatterMaterialQueueIndices<<<blockCount(count), kThreads>>>(
            buffers.keys, count, materialCount, buffers.capacity,
            buffers.counts, buffers.offsets, buffers.cursors,
            buffers.indices, deviceError);
        launchCheck("scatter material queue indices");
    }
}

int downloadAndValidateMaterialQueues(const MaterialQueueBuffers& buffers,
    int materialCount, int incomingCount, std::vector<int>* hostCounts,
    std::vector<int>* hostOffsets)
{
    if (materialCount < 0 || materialCount == std::numeric_limits<int>::max()
            || incomingCount < 0 || hostCounts == nullptr
            || hostOffsets == nullptr)
    {
        throw std::invalid_argument("invalid material queue download request");
    }
    const int keyCount = materialCount + 1;
    hostCounts->resize(static_cast<std::size_t>(keyCount));
    hostOffsets->resize(static_cast<std::size_t>(keyCount));
    const std::size_t bytes = static_cast<std::size_t>(keyCount) * sizeof(int);
    cudaCheck(cudaMemcpy(hostCounts->data(), buffers.counts, bytes,
        cudaMemcpyDeviceToHost), "download material queue counts");
    cudaCheck(cudaMemcpy(hostOffsets->data(), buffers.offsets, bytes,
        cudaMemcpyDeviceToHost), "download material queue offsets");
    int logicalLiveCount = 0;
    cudaCheck(cudaMemcpy(&logicalLiveCount, buffers.liveCount, sizeof(int),
        cudaMemcpyDeviceToHost),
        "download material queue logical live-input count");
    if (logicalLiveCount < 0 || logicalLiveCount > incomingCount)
    {
        throw std::runtime_error(
            "GPU returned an invalid logical live-input count");
    }

    long long prefix = 0;
    for (int key = 0; key < keyCount; ++key)
    {
        const int count = (*hostCounts)[static_cast<std::size_t>(key)];
        const int offset = (*hostOffsets)[static_cast<std::size_t>(key)];
        if (count < 0 || count > buffers.capacity || offset < 0
                || offset > buffers.capacity || offset != prefix)
        {
            throw std::runtime_error(
                "GPU returned an invalid material queue prefix");
        }
        prefix += count;
        if (prefix > buffers.capacity || prefix > incomingCount)
        {
            throw std::runtime_error(
                "GPU returned an overflowing material queue total");
        }
    }
    if (prefix != logicalLiveCount)
    {
        throw std::runtime_error(
            "material queue total does not match logical live inputs");
    }
    return static_cast<int>(prefix);
}

struct QueueSelfCheckResult
{
    bool passed = false;
    std::string detail;
};

QueueSelfCheckResult validateMaterialQueueConstruction()
{
    constexpr int kMaximumPaths = 2145;
    constexpr int kMaximumKeys = kMaximumPaths + 1;
    const int sizes[] = {0, 1, 255, 256, 257, 513, 2145};
    const char* scenarios[] = {"dead", "miss", "uniform", "mixed",
        "unused"};

    DeviceBuffer<RenderPath> paths(kMaximumPaths,
        "allocate queue validation paths");
    DeviceBuffer<Hit> hits(kMaximumPaths,
        "allocate queue validation hits");
    DeviceBuffer<int> flags(kMaximumPaths,
        "allocate queue validation flags");
    DeviceBuffer<int> keys(kMaximumPaths,
        "allocate queue validation keys");
    DeviceBuffer<int> indices(kMaximumPaths,
        "allocate queue validation indices");
    DeviceBuffer<int> counts(kMaximumKeys,
        "allocate queue validation counts");
    DeviceBuffer<int> offsets(kMaximumKeys,
        "allocate queue validation offsets");
    DeviceBuffer<int> cursors(kMaximumKeys,
        "allocate queue validation cursors");
    DeviceBuffer<int> liveCount(1,
        "allocate queue validation live-input count");
    DeviceBuffer<unsigned int> error(1,
        "allocate queue validation error");
    cis565::scan::ScanWorkspace workspace(kMaximumKeys,
        cis565::scan::Config{});

    int completedCases = 0;
    for (const int size : sizes)
    {
        const int materialCount = std::max(7, size);
        const int keyCount = materialCount + 1;
        for (int scenario = 0; scenario < 5; ++scenario)
        {
            std::vector<RenderPath> hostPaths(static_cast<std::size_t>(size));
            std::vector<Hit> hostHits(static_cast<std::size_t>(size));
            std::vector<int> expectedKeys(static_cast<std::size_t>(size), -1);
            std::vector<int> expectedCounts(
                static_cast<std::size_t>(keyCount), 0);
            bool expectsInvalidHit = false;
            for (int index = 0; index < size; ++index)
            {
                RenderPath path;
                path.pixelIndex = index;
                path.alive = 1;
                path.throughput = glm::vec3(1.0f + 0.001f * index);
                path.radiance = glm::vec3(0.002f * index);
                path.etaScale = 1.0 + 0.0001 * index;
                hostPaths[static_cast<std::size_t>(index)] = path;

                Hit hit;
                hit.t = 1.0f + 0.001f * index;
                hit.materialId = 0;
                hit.position = glm::vec3(static_cast<float>(index), 0.0f,
                    -1.0f);
                hit.geometricNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                hit.shadingNormal = hit.geometricNormal;
                hostHits[static_cast<std::size_t>(index)] = hit;

                int expected = 0;
                if (scenario == 0)
                {
                    hostPaths[static_cast<std::size_t>(index)].alive = 0;
                    expected = -1;
                }
                else if (scenario == 1)
                {
                    hostHits[static_cast<std::size_t>(index)].t = -1.0f;
                    expected = materialCount;
                }
                else if (scenario == 2)
                {
                    hostHits[static_cast<std::size_t>(index)].materialId = 2;
                    expected = 2;
                }
                else if (scenario == 3)
                {
                    switch (index % 6)
                    {
                    case 0:
                        hostPaths[static_cast<std::size_t>(index)].alive = 0;
                        expected = -1;
                        break;
                    case 1:
                        hostHits[static_cast<std::size_t>(index)].t = -1.0f;
                        expected = materialCount;
                        break;
                    case 2:
                        expected = 0;
                        break;
                    case 3:
                        hostHits[static_cast<std::size_t>(index)].materialId = 2;
                        expected = 2;
                        break;
                    case 4:
                        hostHits[static_cast<std::size_t>(index)].materialId =
                            materialCount - 1;
                        expected = materialCount - 1;
                        break;
                    default:
                        hostHits[static_cast<std::size_t>(index)].materialId =
                            materialCount + 4;
                        expected = materialCount;
                        expectsInvalidHit = true;
                        break;
                    }
                }
                else
                {
                    const int materialId = index % 2 == 0
                        ? 0 : materialCount - 1;
                    hostHits[static_cast<std::size_t>(index)].materialId =
                        materialId;
                    expected = materialId;
                }
                expectedKeys[static_cast<std::size_t>(index)] = expected;
                if (expected >= 0)
                {
                    ++expectedCounts[static_cast<std::size_t>(expected)];
                }
            }

            if (size > 0)
            {
                cudaCheck(cudaMemcpy(paths.get(), hostPaths.data(),
                    static_cast<std::size_t>(size) * sizeof(RenderPath),
                    cudaMemcpyHostToDevice),
                    "upload queue validation paths");
                cudaCheck(cudaMemcpy(hits.get(), hostHits.data(),
                    static_cast<std::size_t>(size) * sizeof(Hit),
                    cudaMemcpyHostToDevice),
                    "upload queue validation hits");
                cudaCheck(cudaMemset(flags.get(), 0x7f,
                    static_cast<std::size_t>(size) * sizeof(int)),
                    "poison queue validation flags");
                cudaCheck(cudaMemset(indices.get(), 0xff,
                    static_cast<std::size_t>(size) * sizeof(int)),
                    "poison queue validation indices");
            }
            cudaCheck(cudaMemset(error.get(), 0, sizeof(unsigned int)),
                "clear queue validation error");

            const MaterialQueueBuffers buffers{keys.get(), indices.get(),
                counts.get(), offsets.get(), cursors.get(), liveCount.get(),
                size};
            enqueueMaterialQueues(paths.get(), hits.get(), size,
                materialCount, flags.get(), buffers, error.get(), workspace);
            std::vector<int> hostCounts;
            std::vector<int> hostOffsets;
            const int total = downloadAndValidateMaterialQueues(buffers,
                materialCount, size, &hostCounts, &hostOffsets);

            unsigned int errorBits = 0;
            cudaCheck(cudaMemcpy(&errorBits, error.get(), sizeof(errorBits),
                cudaMemcpyDeviceToHost),
                "download queue validation error");
            const unsigned int expectedError = expectsInvalidHit
                ? (1U << 3U) : 0U;
            if ((errorBits & kQueueErrorBit) != 0U
                    || (errorBits & ~kQueueErrorBit) != expectedError)
            {
                std::ostringstream detail;
                detail << "size " << size << ' ' << scenarios[scenario]
                    << " produced error bits 0x" << std::hex << errorBits;
                return {false, detail.str()};
            }
            if (hostCounts != expectedCounts)
            {
                return {false, "queue histogram mismatch at size "
                    + std::to_string(size) + " " + scenarios[scenario]};
            }
            int expectedTotal = 0;
            for (const int expected : expectedKeys)
            {
                expectedTotal += expected >= 0 ? 1 : 0;
            }
            if (total != expectedTotal)
            {
                return {false, "queue live-total mismatch at size "
                    + std::to_string(size) + " " + scenarios[scenario]};
            }

            std::vector<int> hostFlags(static_cast<std::size_t>(size));
            std::vector<int> hostKeys(static_cast<std::size_t>(size));
            std::vector<int> hostIndices(static_cast<std::size_t>(size));
            std::vector<int> hostCursors(static_cast<std::size_t>(keyCount));
            std::vector<RenderPath> roundTripPaths(
                static_cast<std::size_t>(size));
            if (size > 0)
            {
                cudaCheck(cudaMemcpy(hostFlags.data(), flags.get(),
                    static_cast<std::size_t>(size) * sizeof(int),
                    cudaMemcpyDeviceToHost),
                    "download queue validation flags");
                cudaCheck(cudaMemcpy(hostKeys.data(), keys.get(),
                    static_cast<std::size_t>(size) * sizeof(int),
                    cudaMemcpyDeviceToHost),
                    "download queue validation keys");
                cudaCheck(cudaMemcpy(hostIndices.data(), indices.get(),
                    static_cast<std::size_t>(size) * sizeof(int),
                    cudaMemcpyDeviceToHost),
                    "download queue validation indices");
                cudaCheck(cudaMemcpy(roundTripPaths.data(), paths.get(),
                    static_cast<std::size_t>(size) * sizeof(RenderPath),
                    cudaMemcpyDeviceToHost),
                    "download queue validation paths");
            }
            cudaCheck(cudaMemcpy(hostCursors.data(), cursors.get(),
                static_cast<std::size_t>(keyCount) * sizeof(int),
                cudaMemcpyDeviceToHost),
                "download queue validation cursors");
            if (hostKeys != expectedKeys || hostCursors != expectedCounts)
            {
                return {false, "queue classification/cursor mismatch at size "
                    + std::to_string(size) + " " + scenarios[scenario]};
            }
            for (const int flag : hostFlags)
            {
                if (flag != 0)
                {
                    return {false, "queue classification left a live flag set"};
                }
            }
            if (size > 0 && std::memcmp(roundTripPaths.data(),
                    hostPaths.data(), static_cast<std::size_t>(size)
                        * sizeof(RenderPath)) != 0)
            {
                return {false, "queue classification modified a path record"};
            }

            std::vector<int> seen(static_cast<std::size_t>(size), 0);
            for (int key = 0; key < keyCount; ++key)
            {
                const int begin = hostOffsets[static_cast<std::size_t>(key)];
                const int end = begin + hostCounts[static_cast<std::size_t>(key)];
                for (int position = begin; position < end; ++position)
                {
                    const int original = hostIndices[
                        static_cast<std::size_t>(position)];
                    if (original < 0 || original >= size
                            || expectedKeys[static_cast<std::size_t>(original)]
                                != key
                            || ++seen[static_cast<std::size_t>(original)] != 1)
                    {
                        return {false, "queue membership is not unique/exact"};
                    }
                }
            }
            for (int index = 0; index < size; ++index)
            {
                const int expectedSeen = expectedKeys[
                    static_cast<std::size_t>(index)] >= 0 ? 1 : 0;
                if (seen[static_cast<std::size_t>(index)] != expectedSeen)
                {
                    return {false, "queue omitted or duplicated a live input"};
                }
            }
            for (int position = total; position < size; ++position)
            {
                if (hostIndices[static_cast<std::size_t>(position)] != -1)
                {
                    return {false, "queue scatter wrote beyond its live total"};
                }
            }
            ++completedCases;
        }
    }
    return {true, std::to_string(completedCases)
        + " GPU queue cases conserved exact live membership"};
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
        && a.wavefront == b.wavefront && a.rr == b.rr
        && a.rrStart == b.rrStart
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
    if (scene->materials.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max() - 1))
    {
        throw std::overflow_error("scene material count exceeds renderer limits");
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
        allocateDevice(&g_state.queueKeys, pixels,
            "allocate material queue keys");
        allocateDevice(&g_state.queueIndices, pixels,
            "allocate material queue indices");
        g_state.queueKeyCount = static_cast<int>(scene->materials.size()) + 1;
        allocateDevice(&g_state.queueCounts, g_state.queueKeyCount,
            "allocate material queue counts");
        allocateDevice(&g_state.queueOffsets, g_state.queueKeyCount,
            "allocate material queue offsets");
        allocateDevice(&g_state.queueCursors, g_state.queueKeyCount,
            "allocate material queue cursors");
        g_state.queueHostCounts.assign(
            static_cast<std::size_t>(g_state.queueKeyCount), 0);
        g_state.queueHostOffsets.assign(
            static_cast<std::size_t>(g_state.queueKeyCount), 0);
        allocateDevice(&g_state.rrCounters, 2,
            "allocate Russian roulette counters");
        allocateDevice(&g_state.deviceError, 1, "allocate device error flag");
        allocateDevice(&g_state.diagnosticMinimum, 1,
            "allocate BVH diagnostic minimum");
        allocateDevice(&g_state.diagnosticRecord, 1,
            "allocate BVH diagnostic record");
        cudaCheck(cudaMemset(g_state.accumulation, 0,
            static_cast<std::size_t>(pixels) * sizeof(glm::vec3)),
            "clear accumulation image");
        cudaCheck(cudaMemset(g_state.deviceCount, 0, sizeof(int)),
            "initialize device path count");
        const std::size_t queueStateBytes = static_cast<std::size_t>(
            g_state.queueKeyCount) * sizeof(int);
        cudaCheck(cudaMemset(g_state.queueCounts, 0, queueStateBytes),
            "initialize material queue counts");
        cudaCheck(cudaMemset(g_state.queueOffsets, 0, queueStateBytes),
            "initialize material queue offsets");
        cudaCheck(cudaMemset(g_state.queueCursors, 0, queueStateBytes),
            "initialize material queue cursors");
        cudaCheck(cudaMemset(g_state.rrCounters, 0,
            2 * sizeof(unsigned long long)),
            "initialize Russian roulette counters");
        cudaCheck(cudaMemset(g_state.deviceError, 0, sizeof(unsigned int)),
            "initialize renderer device error");
        const unsigned long long initialDiagnosticMinimum = ~0ULL;
        const BvhDiagnostic initialDiagnosticRecord{};
        cudaCheck(cudaMemcpy(g_state.diagnosticMinimum,
            &initialDiagnosticMinimum, sizeof(initialDiagnosticMinimum),
            cudaMemcpyHostToDevice), "initialize BVH diagnostic minimum");
        cudaCheck(cudaMemcpy(g_state.diagnosticRecord,
            &initialDiagnosticRecord, sizeof(initialDiagnosticRecord),
            cudaMemcpyHostToDevice), "initialize BVH diagnostic record");

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

        const std::size_t scanCapacity = std::max<std::size_t>(
            static_cast<std::size_t>(pixels),
            static_cast<std::size_t>(g_state.queueKeyCount));
        g_state.scanWorkspace = std::make_unique<cis565::scan::ScanWorkspace>(
            scanCapacity, cis565::scan::Config{});
        createEventPair(&g_state.intersectionEvents);
        createEventPair(&g_state.sortEvents);
        createEventPair(&g_state.shadeEvents);
        createEventPair(&g_state.compactionEvents);
        createEventPair(&g_state.countEvents);
        createEventPair(&g_state.queueEvents);
        createEventPair(&g_state.diffuseEvents);
        createEventPair(&g_state.mirrorEvents);
        createEventPair(&g_state.dielectricEvents);
        createEventPair(&g_state.emissionEvents);
        createEventPair(&g_state.terminalEvents);
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
    destroyEventPair(&g_state.queueEvents);
    destroyEventPair(&g_state.diffuseEvents);
    destroyEventPair(&g_state.mirrorEvents);
    destroyEventPair(&g_state.dielectricEvents);
    destroyEventPair(&g_state.emissionEvents);
    destroyEventPair(&g_state.terminalEvents);
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
    cudaFree(g_state.queueKeys);
    cudaFree(g_state.queueIndices);
    cudaFree(g_state.queueCounts);
    cudaFree(g_state.queueOffsets);
    cudaFree(g_state.queueCursors);
    cudaFree(g_state.rrCounters);
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
    cudaCheck(cudaMemset(g_state.rrCounters, 0,
        2 * sizeof(unsigned long long)),
        "reset Russian roulette counters");
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

        const int materialCount = static_cast<int>(
            g_state.scene->materials.size());
        if (!g_config.wavefront)
        {
            if (g_config.benchmark)
            {
                beginTimedStage(g_state.sortEvents);
            }
            if (g_config.sort)
            {
                radixSortPairs(currentPaths, scratchPaths, currentHits,
                    scratchHits, launchedPaths, materialCount);
            }
            if (g_config.benchmark)
            {
                row.sortMs = endTimedStage(g_state.sortEvents);
            }
        }

        if (g_config.benchmark)
        {
            cudaCheck(cudaMemset(g_state.rrCounters, 0,
                2 * sizeof(unsigned long long)),
                "clear bounce Russian roulette counters");
        }

        if (!g_config.wavefront)
        {
            if (g_config.benchmark)
            {
                beginTimedStage(g_state.shadeEvents);
            }
            shadePaths<<<blocks, kThreads>>>(g_state.deviceView,
                g_state.materials, materialCount, g_state.lightIndices,
                config, bounce, currentPaths, currentHits, g_state.flags,
                g_state.accumulation, launchedPaths, g_state.deviceError);
            launchCheck("shade path bounce");
            if (g_config.benchmark)
            {
                row.shadeMs = endTimedStage(g_state.shadeEvents);
            }
        }
        else
        {
            const MaterialQueueBuffers queueBuffers{
                g_state.queueKeys, g_state.queueIndices,
                g_state.queueCounts, g_state.queueOffsets,
                g_state.queueCursors, g_state.deviceCount,
                g_state.pixelCount};
            if (g_config.benchmark)
            {
                beginTimedStage(g_state.queueEvents);
            }
            enqueueMaterialQueues(currentPaths, currentHits, launchedPaths,
                materialCount, g_state.flags, queueBuffers,
                g_state.deviceError, *g_state.scanWorkspace);
            if (g_config.benchmark)
            {
                row.queueMs = endTimedStage(g_state.queueEvents);
            }

            const auto downloadStart = std::chrono::steady_clock::now();
            const int liveInputCount = downloadAndValidateMaterialQueues(
                queueBuffers, materialCount, launchedPaths,
                &g_state.queueHostCounts, &g_state.queueHostOffsets);
            const auto downloadEnd = std::chrono::steady_clock::now();
            if ((g_config.compact || g_config.benchmark || bounce == 0)
                    && liveInputCount != logicalActive)
            {
                throw std::runtime_error(
                    "material queues do not conserve the logical live input");
            }
            logicalActive = liveInputCount;
            row.activeBefore = liveInputCount;
            if (g_config.benchmark)
            {
                row.queueHostMs += std::chrono::duration<double, std::milli>(
                    downloadEnd - downloadStart).count();
            }

            for (int materialId = 0; materialId < materialCount;
                    ++materialId)
            {
                const int queueCount = g_state.queueHostCounts[
                    static_cast<std::size_t>(materialId)];
                if (queueCount == 0)
                {
                    continue;
                }
                const int queueOffset = g_state.queueHostOffsets[
                    static_cast<std::size_t>(materialId)];
                const int queueBlocks = blockCount(queueCount);
                const Material& material = g_state.scene->materials[
                    static_cast<std::size_t>(materialId)];
                ++row.kernelDispatches;

                if (material.emittance > 0.0f)
                {
                    row.queuedEmission += queueCount;
                    if (g_config.benchmark)
                    {
                        beginTimedStage(g_state.emissionEvents);
                    }
                    const auto dispatchStart = std::chrono::steady_clock::now();
                    shadeEmissionQueue<<<queueBlocks, kThreads>>>(
                        g_state.deviceView, g_state.materials, materialCount,
                        config, currentPaths, currentHits, g_state.flags,
                        g_state.accumulation, g_state.queueIndices,
                        queueOffset, queueCount, launchedPaths,
                        g_state.deviceError);
                    launchCheck("shade emission material queue");
                    const auto dispatchEnd = std::chrono::steady_clock::now();
                    if (g_config.benchmark)
                    {
                        row.queueHostMs += std::chrono::duration<double,
                            std::milli>(dispatchEnd - dispatchStart).count();
                        row.emissionMs += endTimedStage(
                            g_state.emissionEvents);
                    }
                }
                else if (material.hasRefractive > 0.0f)
                {
                    row.queuedDielectric += queueCount;
                    if (g_config.benchmark)
                    {
                        beginTimedStage(g_state.dielectricEvents);
                    }
                    const auto dispatchStart = std::chrono::steady_clock::now();
                    shadeDielectricQueue<<<queueBlocks, kThreads>>>(
                        g_state.deviceView, g_state.materials, materialCount,
                        config, bounce, currentPaths, currentHits,
                        g_state.flags, g_state.accumulation,
                        g_state.queueIndices, queueOffset, queueCount,
                        launchedPaths, g_state.deviceError);
                    launchCheck("shade dielectric material queue");
                    const auto dispatchEnd = std::chrono::steady_clock::now();
                    if (g_config.benchmark)
                    {
                        row.queueHostMs += std::chrono::duration<double,
                            std::milli>(dispatchEnd - dispatchStart).count();
                        row.dielectricMs += endTimedStage(
                            g_state.dielectricEvents);
                    }
                }
                else if (material.hasReflective > 0.0f)
                {
                    row.queuedMirror += queueCount;
                    if (g_config.benchmark)
                    {
                        beginTimedStage(g_state.mirrorEvents);
                    }
                    const auto dispatchStart = std::chrono::steady_clock::now();
                    shadeMirrorQueue<<<queueBlocks, kThreads>>>(
                        g_state.deviceView, g_state.materials, materialCount,
                        config, bounce, currentPaths, currentHits,
                        g_state.flags, g_state.accumulation,
                        g_state.queueIndices, queueOffset, queueCount,
                        launchedPaths, g_state.deviceError);
                    launchCheck("shade mirror material queue");
                    const auto dispatchEnd = std::chrono::steady_clock::now();
                    if (g_config.benchmark)
                    {
                        row.queueHostMs += std::chrono::duration<double,
                            std::milli>(dispatchEnd - dispatchStart).count();
                        row.mirrorMs += endTimedStage(g_state.mirrorEvents);
                    }
                }
                else
                {
                    row.queuedDiffuse += queueCount;
                    if (g_config.benchmark)
                    {
                        beginTimedStage(g_state.diffuseEvents);
                    }
                    const auto dispatchStart = std::chrono::steady_clock::now();
                    shadeDiffuseQueue<<<queueBlocks, kThreads>>>(
                        g_state.deviceView, g_state.materials, materialCount,
                        g_state.lightIndices, config, bounce, currentPaths,
                        currentHits, g_state.flags, g_state.accumulation,
                        g_state.queueIndices, queueOffset, queueCount,
                        launchedPaths, g_state.deviceError);
                    launchCheck("shade diffuse material queue");
                    const auto dispatchEnd = std::chrono::steady_clock::now();
                    if (g_config.benchmark)
                    {
                        row.queueHostMs += std::chrono::duration<double,
                            std::milli>(dispatchEnd - dispatchStart).count();
                        row.diffuseMs += endTimedStage(g_state.diffuseEvents);
                    }
                }
            }

            const int terminalCount = g_state.queueHostCounts[
                static_cast<std::size_t>(materialCount)];
            if (terminalCount > 0)
            {
                const int terminalOffset = g_state.queueHostOffsets[
                    static_cast<std::size_t>(materialCount)];
                row.queuedTerminal = terminalCount;
                ++row.kernelDispatches;
                if (g_config.benchmark)
                {
                    beginTimedStage(g_state.terminalEvents);
                }
                const auto dispatchStart = std::chrono::steady_clock::now();
                shadeTerminalQueue<<<blockCount(terminalCount), kThreads>>>(
                    materialCount, currentPaths, currentHits, g_state.flags,
                    g_state.accumulation, g_state.queueIndices,
                    terminalOffset, terminalCount, launchedPaths,
                    g_state.deviceError);
                launchCheck("shade terminal material queue");
                const auto dispatchEnd = std::chrono::steady_clock::now();
                if (g_config.benchmark)
                {
                    row.queueHostMs += std::chrono::duration<double,
                        std::milli>(dispatchEnd - dispatchStart).count();
                    row.terminalMs += endTimedStage(g_state.terminalEvents);
                }
            }
            const long long scheduled = static_cast<long long>(
                row.queuedDiffuse) + row.queuedMirror
                + row.queuedDielectric + row.queuedEmission
                + row.queuedTerminal;
            if (scheduled != liveInputCount)
            {
                throw std::runtime_error(
                    "material queue dispatch does not conserve live inputs");
            }
            row.shadeMs = row.diffuseMs + row.mirrorMs + row.dielectricMs
                + row.emissionMs + row.terminalMs;
        }

        if (g_config.benchmark)
        {
            unsigned long long rrCounts[2] = {0ULL, 0ULL};
            cudaCheck(cudaMemcpy(rrCounts, g_state.rrCounters,
                sizeof(rrCounts), cudaMemcpyDeviceToHost),
                "download Russian roulette diagnostics");
            if (rrCounts[1] > rrCounts[0]
                    || rrCounts[0] > static_cast<unsigned long long>(
                        launchedPaths))
            {
                throw std::runtime_error(
                    "GPU returned invalid Russian roulette diagnostics");
            }
            row.rrTested = rrCounts[0];
            row.rrTerminated = rrCounts[1];
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
    const std::vector<int> savedQueueHostCounts = g_state.queueHostCounts;
    const std::vector<int> savedQueueHostOffsets = g_state.queueHostOffsets;
    const int savedGuiTracedDepth = g_guiData != nullptr
        ? g_guiData->TracedDepth : 0;
    const int savedDiagnosticMode = g_state.diagnosticMode;
    const unsigned long long savedDiagnosticTarget = g_state.diagnosticTarget;
    unsigned long long savedDiagnosticMinimum = ~0ULL;
    BvhDiagnostic savedDiagnosticRecord;
    unsigned long long savedRrCounters[2] = {0ULL, 0ULL};
    unsigned int savedDeviceError = 0U;
    int savedDeviceCount = 0;
    std::vector<int> savedQueueCounts(
        static_cast<std::size_t>(g_state.queueKeyCount));
    std::vector<int> savedQueueOffsets(
        static_cast<std::size_t>(g_state.queueKeyCount));
    std::vector<int> savedQueueCursors(
        static_cast<std::size_t>(g_state.queueKeyCount));
    cudaCheck(cudaMemcpy(&savedDiagnosticMinimum, g_state.diagnosticMinimum,
        sizeof(savedDiagnosticMinimum), cudaMemcpyDeviceToHost),
        "save BVH diagnostic minimum");
    cudaCheck(cudaMemcpy(&savedDiagnosticRecord, g_state.diagnosticRecord,
        sizeof(savedDiagnosticRecord), cudaMemcpyDeviceToHost),
        "save BVH diagnostic record");
    cudaCheck(cudaMemcpy(savedRrCounters, g_state.rrCounters,
        sizeof(savedRrCounters), cudaMemcpyDeviceToHost),
        "save Russian roulette counters");
    cudaCheck(cudaMemcpy(&savedDeviceError, g_state.deviceError,
        sizeof(savedDeviceError), cudaMemcpyDeviceToHost),
        "save renderer device error");
    cudaCheck(cudaMemcpy(&savedDeviceCount, g_state.deviceCount,
        sizeof(savedDeviceCount), cudaMemcpyDeviceToHost),
        "save renderer device count");
    const std::size_t savedQueueBytes = savedQueueCounts.size() * sizeof(int);
    cudaCheck(cudaMemcpy(savedQueueCounts.data(), g_state.queueCounts,
        savedQueueBytes, cudaMemcpyDeviceToHost),
        "save material queue counts");
    cudaCheck(cudaMemcpy(savedQueueOffsets.data(), g_state.queueOffsets,
        savedQueueBytes, cudaMemcpyDeviceToHost),
        "save material queue offsets");
    cudaCheck(cudaMemcpy(savedQueueCursors.data(), g_state.queueCursors,
        savedQueueBytes, cudaMemcpyDeviceToHost),
        "save material queue cursors");
    std::vector<glm::vec3> savedAccumulation;
    pathtraceDownloadAccumulation(savedAccumulation);

    const auto addCheck = [&](const std::string& name, bool passed,
                              const std::string& detail)
    {
        report.checks.push_back(RendererValidationCheck{name, passed, detail});
    };
    const auto restore = [&]()
    {
        cudaCheck(cudaMemcpy(g_state.diagnosticMinimum,
            &savedDiagnosticMinimum, sizeof(savedDiagnosticMinimum),
            cudaMemcpyHostToDevice), "restore BVH diagnostic minimum");
        cudaCheck(cudaMemcpy(g_state.diagnosticRecord, &savedDiagnosticRecord,
            sizeof(savedDiagnosticRecord), cudaMemcpyHostToDevice),
            "restore BVH diagnostic record");
        cudaCheck(cudaMemcpy(g_state.rrCounters, savedRrCounters,
            sizeof(savedRrCounters), cudaMemcpyHostToDevice),
            "restore Russian roulette counters");
        cudaCheck(cudaMemcpy(g_state.deviceError, &savedDeviceError,
            sizeof(savedDeviceError), cudaMemcpyHostToDevice),
            "restore renderer device error");
        cudaCheck(cudaMemcpy(g_state.deviceCount, &savedDeviceCount,
            sizeof(savedDeviceCount), cudaMemcpyHostToDevice),
            "restore renderer device count");
        cudaCheck(cudaMemcpy(g_state.queueCounts, savedQueueCounts.data(),
            savedQueueBytes, cudaMemcpyHostToDevice),
            "restore material queue counts");
        cudaCheck(cudaMemcpy(g_state.queueOffsets, savedQueueOffsets.data(),
            savedQueueBytes, cudaMemcpyHostToDevice),
            "restore material queue offsets");
        cudaCheck(cudaMemcpy(g_state.queueCursors, savedQueueCursors.data(),
            savedQueueBytes, cudaMemcpyHostToDevice),
            "restore material queue cursors");
        g_state.diagnosticMode = savedDiagnosticMode;
        g_state.diagnosticTarget = savedDiagnosticTarget;
        g_config = savedConfig;
        pathtraceUploadAccumulation(savedAccumulation, savedSamples);
        g_state.benchmarkRows = savedRows;
        g_state.benchmarkSummary = savedSummary;
        g_state.queueHostCounts = savedQueueHostCounts;
        g_state.queueHostOffsets = savedQueueHostOffsets;
        if (g_guiData != nullptr)
        {
            g_guiData->TracedDepth = savedGuiTracedDepth;
        }
    };

    try
    {
        bool rrBoundsValidated = true;
        for (const int invalidStart : {0, 4097})
        {
            RendererConfig invalid = savedConfig;
            invalid.rr = false;
            invalid.rrStart = invalidStart;
            try
            {
                validateConfig(invalid);
                rrBoundsValidated = false;
            }
            catch (const std::invalid_argument&)
            {
            }
        }
        RendererConfig validLargeStart = savedConfig;
        validLargeStart.sort = false;
        validLargeStart.rr = false;
        validLargeStart.maxDepth = 1;
        validLargeStart.rrStart = 4096;
        try
        {
            validateConfig(validLargeStart);
        }
        catch (...)
        {
            rrBoundsValidated = false;
        }
        addCheck("rr_config_bounds", rrBoundsValidated,
            "rrStart is always validated in [1,4096] and may exceed maxDepth");

        bool waveSortRejected = false;
        RendererConfig invalidWaveSort = savedConfig;
        invalidWaveSort.wavefront = true;
        invalidWaveSort.sort = true;
        try
        {
            validateConfig(invalidWaveSort);
        }
        catch (const std::invalid_argument& error)
        {
            waveSortRejected = std::string(error.what())
                == "--sort on requires --wavefront off";
        }
        addCheck("wavefront_sort_rejection", waveSortRejected,
            "wavefront plus material sort returns the actionable CLI error");

        RendererConfig referenceConfig = savedConfig;
        referenceConfig.benchmark = false;
        referenceConfig.compact = true;
        referenceConfig.sort = false;
        referenceConfig.wavefront = false;
        referenceConfig.rr = false;
        referenceConfig.rrStart = 3;
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
        candidateConfig.wavefront = true;
        const std::vector<glm::vec3> wavefront = renderOne(candidateConfig);
        addCheck("wavefront_equivalence",
            exactAccumulation(reference, wavefront),
            "raw float sums for unified and exact-material queue scheduling");

        candidateConfig.compact = false;
        const std::vector<glm::vec3> wavefrontUncompacted = renderOne(
            candidateConfig);
        addCheck("wavefront_uncompacted_equivalence",
            exactAccumulation(reference, wavefrontUncompacted),
            "queue scheduling preserves raw sums without path compaction");

        RendererConfig depthOneLegacy = referenceConfig;
        depthOneLegacy.maxDepth = 1;
        depthOneLegacy.rr = false;
        const std::vector<glm::vec3> depthOneReference = renderOne(
            depthOneLegacy);
        RendererConfig depthOneWavefront = depthOneLegacy;
        depthOneWavefront.wavefront = true;
        const std::vector<glm::vec3> depthOneCandidate = renderOne(
            depthOneWavefront);
        addCheck("wavefront_depth_one_equivalence",
            exactAccumulation(depthOneReference, depthOneCandidate),
            "emission-before-depth-cutoff raw equality at depth one");

        RendererConfig depthOneRrLegacy = depthOneLegacy;
        depthOneRrLegacy.rr = true;
        depthOneRrLegacy.rrStart = 1;
        const std::vector<glm::vec3> depthOneRrLegacyResult = renderOne(
            depthOneRrLegacy);
        RendererConfig depthOneRrWavefront = depthOneRrLegacy;
        depthOneRrWavefront.wavefront = true;
        const std::vector<glm::vec3> depthOneRrWavefrontResult = renderOne(
            depthOneRrWavefront);
        addCheck("rr_ineligible_scene_equivalence",
            exactAccumulation(depthOneReference, depthOneRrLegacyResult)
                && exactAccumulation(depthOneCandidate,
                    depthOneRrWavefrontResult),
            "RR on/off is raw-identical in both schedulers when depth <= rrStart");

        RendererConfig debugLegacy = referenceConfig;
        debugLegacy.debug = RendererDebugMode::Normals;
        debugLegacy.rr = false;
        const std::vector<glm::vec3> debugReference = renderOne(debugLegacy);
        RendererConfig debugWavefront = debugLegacy;
        debugWavefront.wavefront = true;
        const std::vector<glm::vec3> debugCandidate = renderOne(debugWavefront);
        addCheck("wavefront_debug_equivalence",
            exactAccumulation(debugReference, debugCandidate),
            "first-hit debug mode is raw-identical across schedulers");

        RendererConfig rrLegacy = referenceConfig;
        rrLegacy.rr = true;
        rrLegacy.rrStart = 1;
        rrLegacy.maxDepth = std::max(2, rrLegacy.maxDepth);
        const std::vector<glm::vec3> rrReference = renderOne(rrLegacy);
        RendererConfig rrWavefront = rrLegacy;
        rrWavefront.wavefront = true;
        const std::vector<glm::vec3> rrCandidate = renderOne(rrWavefront);
        addCheck("wavefront_rr_equivalence",
            exactAccumulation(rrReference, rrCandidate),
            "raw float sums use identical RR and original-pixel RNG keys");

        RendererConfig uninstrumented = rrWavefront;
        uninstrumented.benchmark = false;
        const std::vector<glm::vec3> uninstrumentedResult = renderOne(
            uninstrumented);
        RendererConfig instrumented = uninstrumented;
        instrumented.benchmark = true;
        const std::vector<glm::vec3> instrumentedResult = renderOne(
            instrumented);
        addCheck("instrumentation_equivalence",
            exactAccumulation(uninstrumentedResult, instrumentedResult),
            "benchmark counters and timing do not change sampled transport");

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

        const QueueSelfCheckResult queueSelfCheck =
            validateMaterialQueueConstruction();
        addCheck("wavefront_queue_conservation", queueSelfCheck.passed,
            queueSelfCheck.detail);

        DeviceBuffer<int> deviceFailures(1, "allocate validation result");
        int failures = 0;
        rendererUnitValidation<<<1, 1>>>(deviceFailures.get());
        launchCheck("run renderer GPU unit validation");
        cudaCheck(cudaMemcpy(&failures, deviceFailures.get(), sizeof(failures),
            cudaMemcpyDeviceToHost), "download GPU validation result");
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

        DeviceBuffer<int> proceduralFailuresDevice(1,
            "allocate procedural texture validation result");
        proceduralTextureUnitValidation<<<1, 1>>>(
            proceduralFailuresDevice.get());
        launchCheck("run procedural texture GPU unit validation");
        int proceduralFailures = 0;
        cudaCheck(cudaMemcpy(&proceduralFailures,
            proceduralFailuresDevice.get(), sizeof(proceduralFailures),
            cudaMemcpyDeviceToHost),
            "download procedural texture validation result");
        addCheck("procedural_marble_known_sample",
            (proceduralFailures & 0x1) == 0,
            "marble ID 2 matches the authored UV(0,0) RGB sample");
        addCheck("procedural_cells_known_samples",
            (proceduralFailures & 0x2) == 0,
            "cells ID 3 matches authored grout and gold-center RGB samples");
        addCheck("procedural_dense_range_distinct",
            (proceduralFailures & 0x4) == 0,
            "dense varied-UV samples are finite, unit-range, and distinguish both IDs");
        addCheck("procedural_exact_integer_periodicity",
            (proceduralFailures & 0x8) == 0,
            "dyadic UV samples repeat bit-exactly after integer (-2,+3) offsets");
        addCheck("procedural_surface_multiplication",
            (proceduralFailures & 0x10) == 0,
            "evaluateSurface multiplies linear RGB by IDs 2/3 without checker consumption");

        DeviceBuffer<int> rrFailuresDevice(1,
            "allocate Russian roulette validation result");
        DeviceBuffer<unsigned long long> rrDiagnosticsDevice(2,
            "allocate Russian roulette validation diagnostics");
        russianRouletteUnitValidation<<<1, 1>>>(rrFailuresDevice.get(),
            rrDiagnosticsDevice.get());
        launchCheck("run Russian roulette GPU unit validation");
        int rrFailures = 0;
        cudaCheck(cudaMemcpy(&rrFailures, rrFailuresDevice.get(),
            sizeof(rrFailures), cudaMemcpyDeviceToHost),
            "download Russian roulette validation result");
        addCheck("rr_probability_clamps", (rrFailures & 0x1) == 0,
            "actual guarded probability helper clamps corrected beta to [.05,.95]");
        addCheck("rr_eta_compensation", (rrFailures & 0x2) == 0,
            "successful refraction beta and inverse eta scale cancel for RR probability");
        addCheck("rr_deterministic_outcomes", (rrFailures & 0x4) == 0,
            "dimension-32 tuples deterministically exercise survival and termination counters");
        addCheck("rr_future_throughput_only", (rrFailures & 0x8) == 0,
            "survival rescales only future throughput and preserves expected beta");
        addCheck("rr_eligibility", (rrFailures & 0x10) == 0,
            "roulette leaves paths and diagnostics untouched before rrStart");
        addCheck("rr_terminal_exclusions", (rrFailures & 0x20) == 0,
            "terminal, debug, and final-depth completion leave RR counters at zero");
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
