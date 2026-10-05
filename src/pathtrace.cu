#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <vector>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define ANTI_ALIASING 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

// Timing stuff for the kernels; mimicked after Project 2's timer code
struct EventPair  // stores the start and stop times
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    void create()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void destroy()
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
            start = nullptr;
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
            stop = nullptr;
        }
    }

    void recordStart() const
    {
        cudaEventRecord(start);
    }

    void recordStop() const
    {
        cudaEventRecord(stop);
    }

    float elapsedMs() const
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

struct Profiler
{
    bool initialized = false;
    EventPair generate;
    EventPair gather;
    std::vector<EventPair> intersections;
    std::vector<EventPair> sorts;
    std::vector<EventPair> shades;
    std::vector<EventPair> compactions;

    void init(int maxDepth)
    {
        destroy(); // reuse

        generate.create();
        gather.create();
        intersections.resize(maxDepth);
        sorts.resize(maxDepth);
        shades.resize(maxDepth);
        compactions.resize(maxDepth);

        for (int i = 0; i < maxDepth; i++)
        {
            intersections[i].create();
            sorts[i].create();
            shades[i].create();
            compactions[i].create();
        }
        initialized = true;
    }

    void destroy()
    {
        if (!initialized) return;

        generate.destroy();
        gather.destroy();
        for (EventPair& eventPair : intersections) eventPair.destroy();
        for (EventPair& eventPair : sorts) eventPair.destroy();
        for (EventPair& eventPair : shades) eventPair.destroy();
        for (EventPair& eventPair : compactions) eventPair.destroy();
        intersections.clear();
        sorts.clear();
        shades.clear();
        compactions.clear();
        initialized = false;
    }

    // Show results in the gui window
    void writeResults(
        GuiDataContainer* data,
        int tracedDepth,
        bool sortEnabled,
        bool compactionEnabled) const
    {
        if (!initialized || data == nullptr) return;

        data->GenerateRayMs = generate.elapsedMs();
        data->FinalGatherMs = gather.elapsedMs();
        data->ComputeIntersectionsMs = 0.0f;
        data->SortMs = 0.0f;
        data->ShadeMs = 0.0f;
        data->CompactMs = 0.0f;

        for (int depth = 0; depth < tracedDepth; ++depth)
        {
            data->ComputeIntersectionsMs += intersections[depth].elapsedMs();
            data->ShadeMs += shades[depth].elapsedMs();
            if (sortEnabled)
            {
                data->SortMs += sorts[depth].elapsedMs();
            }
            if (compactionEnabled)
            {
                data->CompactMs += compactions[depth].elapsedMs();
            }
        }
    }
};


Profiler profiler;  // this will be used for timing


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* dev_image = nullptr;
static Geom* dev_geoms = nullptr;
static Triangle* dev_triangles = nullptr;
static Primitive* dev_primitives = nullptr;
static BvhNode* dev_bvhNodes = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;
static int* dev_materialKeys = nullptr;
static glm::vec3* dev_iterationImage = nullptr;
static cudaArray_t dev_environmentArray = nullptr;
static cudaTextureObject_t dev_environmentTexture = 0;


__host__ __device__ bool isFiniteVector(const glm::vec3& v)
{
    return v.x == v.x && v.y == v.y && v.z == v.z
        && abs(v.x) < FLT_MAX && abs(v.y) < FLT_MAX && abs(v.z) < FLT_MAX;
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    if (!scene->geoms.empty())
    {
        cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
        cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    }

    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    if (!scene->primitives.empty())
    {
        cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
        cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    }

    if (!scene->bvhNodes.empty())
    {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BvhNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_iterationImage, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_iterationImage, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));

    if (!scene->environment.pixels.empty())
    {
        std::vector<float4> environmentPixels(scene->environment.pixels.size());
        for (size_t i = 0; i < scene->environment.pixels.size(); i++)
        {
            const glm::vec3& pixel = scene->environment.pixels[i];
            environmentPixels[i] = make_float4(pixel.x, pixel.y, pixel.z, 1.0f);
        }

        cudaChannelFormatDesc channelDescription = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&dev_environmentArray, &channelDescription,
            scene->environment.width, scene->environment.height);
        cudaMemcpy2DToArray(dev_environmentArray, 0, 0,
            environmentPixels.data(), scene->environment.width * sizeof(float4),
            scene->environment.width * sizeof(float4), scene->environment.height,
            cudaMemcpyHostToDevice);

        cudaResourceDesc resourceDescription{};
        resourceDescription.resType = cudaResourceTypeArray;
        resourceDescription.res.array.array = dev_environmentArray;
        cudaTextureDesc textureDescription{};
        textureDescription.addressMode[0] = cudaAddressModeWrap;
        textureDescription.addressMode[1] = cudaAddressModeClamp;
        textureDescription.filterMode = cudaFilterModeLinear;
        textureDescription.readMode = cudaReadModeElementType;
        textureDescription.normalizedCoords = 1;
        cudaCreateTextureObject(&dev_environmentTexture, &resourceDescription, &textureDescription, nullptr);
    }

    profiler.init(hst_scene->state.traceDepth);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_triangles);
    cudaFree(dev_primitives);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_iterationImage);
    cudaFree(dev_materialKeys);
    if (dev_environmentTexture != 0) cudaDestroyTextureObject(dev_environmentTexture);
    if (dev_environmentArray != nullptr) cudaFreeArray(dev_environmentArray);

    dev_image = nullptr;
    dev_iterationImage = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_triangles = nullptr;
    dev_primitives = nullptr;
    dev_bvhNodes = nullptr;
    dev_materials = nullptr;
    dev_intersections = nullptr;
    dev_materialKeys = nullptr;
    dev_environmentTexture = 0;
    dev_environmentArray = nullptr;
    hst_scene = nullptr;

    profiler.destroy();

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__host__ __device__ glm::vec2 sampleConcentricDisk(float randomX, float randomY)
{
    float x = 2.0f * randomX - 1.0f;
    float y = 2.0f * randomY - 1.0f;
    if (x == 0.0f && y == 0.0f) return glm::vec2(0.0f);

    float radius;
    float angle;
    if (abs(x) > abs(y))
    {
        radius = x;
        angle = PI * 0.25f * (y / x);
    }
    else
    {
        radius = y;
        angle = PI * 0.5f - PI * 0.25f * (x / y);
    }
    return radius * glm::vec2(cos(angle), sin(angle));
}

__global__ void generateRayFromCamera(
    Camera cam,
    int iter,
    int traceDepth,
    bool enableDepthOfField,
    PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

#if ANTI_ALIASING
        // Anti-aliasing
        float newX = static_cast<float>(x) + uniform01(rng);
        float newY = static_cast<float>(y) + uniform01(rng);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (newX - static_cast<float>(cam.resolution.x) * 0.5f)
            - cam.up * cam.pixelLength.y * (newY - static_cast<float>(cam.resolution.y) * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif // ANTI_ALIASING

        if (enableDepthOfField && cam.apertureRadius > 0.0f && cam.focusDistance > 0.0f)
        {
            float focusT = cam.focusDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 focusPoint = cam.position + focusT * segment.ray.direction;
            glm::vec2 lensSample = sampleConcentricDisk(uniform01(rng), uniform01(rng));
            glm::vec3 lensOffset = cam.apertureRadius * (lensSample.x * cam.right + lensSample.y * cam.up);
            segment.ray.origin = cam.position + lensOffset;
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.currentIor = 1.0f;
    }
}

/**
* Compute closest intersections only.
*/
__device__ void testPrimitive(
    int primitiveIndex,
    const Primitive* primitives,
    const Geom* geoms,
    const Triangle* triangles,
    Ray ray,
    float& closestT,
    glm::vec3& closestNormal,
    bool& closestOutside,
    int& closestPrimitive)
{
    glm::vec3 normal(0.0f);
    bool outside = true;
    float t = primitiveIntersectionTest(
        primitives[primitiveIndex], geoms, triangles, ray, normal, outside);
    if (t > 0.0f && t < closestT)
    {
        closestT = t;
        closestPrimitive = primitiveIndex;
        closestNormal = normal;
        closestOutside = outside;
    }
}

__global__ void computeIntersections(
    int numPaths,
    PathSegment* pathSegments,
    const Geom* geoms,
    const Triangle* triangles,
    const Primitive* primitives,
    int primitiveCount,
    const BvhNode* bvhNodes,
    int bvhNodeCount,
    bool useBvh,
    int missMaterialKey,
    ShadeableIntersection* intersections,
    int* materialKeys)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numPaths) return;

    PathSegment path = pathSegments[pathIndex];
    ShadeableIntersection result{};
    result.t = -1.0f;
    result.materialId = -1;
    result.geomId = -1;
    result.outside = true;

    if (path.remainingBounces <= 0)
    {
        intersections[pathIndex] = result;
        materialKeys[pathIndex] = missMaterialKey + 1; // makes the path terminated for removal
        return;
    }

    float closestT = FLT_MAX;
    glm::vec3 closestNormal(0.0f);
    bool closestOutside = true;
    int hitPrimitiveIndex = -1;

    bool traversalOverflow = false;
    if (useBvh && bvhNodeCount > 0)
    {
        int stack[64];
        int stackSize = 1;
        stack[0] = 0;
        while (stackSize > 0)
        {
            int nodeIndex = stack[--stackSize];
            const BvhNode& node = bvhNodes[nodeIndex];
            Aabb nodeBounds{ node.boundsMin, node.boundsMax };
            float nodeNear = 0.0f;
            if (!aabbIntersectionTest(nodeBounds, path.ray, closestT, nodeNear)) continue;

            if (node.primitiveCount > 0)
            {
                for (int i = 0; i < node.primitiveCount; i++)
                {
                    testPrimitive(node.leftmostChild + i, primitives, geoms, triangles, path.ray,
                        closestT, closestNormal, closestOutside, hitPrimitiveIndex);
                }
            }
            else
            {
                const BvhNode& left = bvhNodes[node.leftmostChild];
                const BvhNode& right = bvhNodes[node.rightChild];
                Aabb leftBounds{ left.boundsMin, left.boundsMax };
                Aabb rightBounds{ right.boundsMin, right.boundsMax };
                float leftNear = 0.0f;
                float rightNear = 0.0f;
                bool hitLeft = aabbIntersectionTest(leftBounds, path.ray, closestT, leftNear);
                bool hitRight = aabbIntersectionTest(rightBounds, path.ray, closestT, rightNear);

                if (stackSize + static_cast<int>(hitLeft) + static_cast<int>(hitRight) > 64)
                {
                    traversalOverflow = true;
                    break;
                }
                if (hitLeft && hitRight)
                {
                    if (leftNear < rightNear)
                    {
                        stack[stackSize++] = node.rightChild;
                        stack[stackSize++] = node.leftmostChild;
                    }
                    else
                    {
                        stack[stackSize++] = node.leftmostChild;
                        stack[stackSize++] = node.rightChild;
                    }
                }
                else if (hitLeft)
                {
                    stack[stackSize++] = node.leftmostChild;
                }
                else if (hitRight)
                {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    }

    if (!useBvh || bvhNodeCount == 0 || traversalOverflow)
    {
        for (int i = 0; i < primitiveCount; i++)
        {
            testPrimitive(i, primitives, geoms, triangles, path.ray,
                closestT, closestNormal, closestOutside, hitPrimitiveIndex);
        }
    }

    if (hitPrimitiveIndex >= 0)
    {
        // The ray hits something
        if (glm::dot(closestNormal, path.ray.direction) > 0.0f)
        {
            closestNormal = -closestNormal;  // flipped normal if hitting backface
        }

        result.t = closestT;
        result.materialId = primitives[hitPrimitiveIndex].materialId;
        result.geomId = hitPrimitiveIndex;
        result.surfaceNormal = closestNormal;
        result.outside = closestOutside;
        materialKeys[pathIndex] = result.materialId;
    }
    else
    {
        materialKeys[pathIndex] = missMaterialKey;
    }

    intersections[pathIndex] = result;
}

/**
* Shade if hitting a light or scatter the ray otherwise.
*/
__device__ glm::vec3 sampleEnvironment(
    cudaTextureObject_t environmentTexture,
    glm::vec3 direction,
    float intensity,
    float rotation)
{
    direction = glm::normalize(direction);
    float u = atan2f(direction.z, direction.x) / TWO_PI + 0.5f + rotation / 360.0f;
    float v = acosf(glm::clamp(direction.y, -1.0f, 1.0f)) / PI;
    float4 pixel = tex2D<float4>(environmentTexture, u, v);
    return glm::vec3(pixel.x, pixel.y, pixel.z) * intensity;
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int numPaths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t environmentTexture,
    bool enableEnvironment,
    float environmentIntensity,
    float environmentRotation,
    glm::vec3* iterationImage)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0.0f) // no intersection
    {
        glm::vec3 background = BACKGROUND_COLOR;
        if (enableEnvironment && environmentTexture != 0)
        {
            background = sampleEnvironment(
                environmentTexture, path.ray.direction, environmentIntensity, environmentRotation);
        }
        iterationImage[path.pixelIndex] += path.throughput * background;
        path.remainingBounces = 0;  // terminate path
        return;
    }

    // If the intersection exists...
    Material material = materials[intersection.materialId];
    if (material.emittance > 0.0f)
    {
        iterationImage[path.pixelIndex] += path.throughput * material.color * material.emittance;
        path.remainingBounces = 0;
        return;
    }

    path.remainingBounces--;
    if (path.remainingBounces <= 0) return;  // does not hit light and no more bounces left

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth + 1);
    glm::vec3 hitPoint = path.ray.origin + intersection.t * glm::normalize(path.ray.direction);
    scatterRay(path, hitPoint, intersection.surfaceNormal, intersection.outside, material, rng);

    // Just a sanity check to make sure the path is still valid
    if (!isFiniteVector(path.ray.origin)
        || !isFiniteVector(path.ray.direction)
        || !isFiniteVector(path.throughput)
        || (path.throughput.x <= 0.0f
            && path.throughput.y <= 0.0f
            && path.throughput.z <= 0.0f))
    {
        path.remainingBounces = 0;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, const glm::vec3* iterationImage)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        image[index] += iterationImage[index];
    }
}

struct IsPathTerminated  // used for thrust::remove_if
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces <= 0;
    }
};

void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int maxTraceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelCount = cam.resolution.x * cam.resolution.y;
    const int materialCount = static_cast<int>(hst_scene->materials.size());
    const bool enableCompaction = guiData == nullptr ? hst_scene->enableCompaction : guiData->EnableCompaction;
    const bool sortByMaterial = guiData == nullptr ? hst_scene->sortByMaterial : guiData->SortByMaterial;
    const bool useBvh = guiData == nullptr ? hst_scene->useBvh : guiData->UseBvh;
    const bool enableDepthOfField = guiData == nullptr ? hst_scene->enableDepthOfField : guiData->EnableDepthOfField;
    const bool enableEnvironment = guiData == nullptr ? hst_scene->enableEnvironment : guiData->EnableEnvironment;

    if (guiData != nullptr)
    {
        guiData->TracedDepth = 0;
        guiData->ActivePathsByDepth.clear();
        guiData->ActivePathsByDepth.reserve(maxTraceDepth);
    }

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    
    // 1D block for path tracing
    const int blockSize1d = 128;

    cudaMemset(dev_iterationImage, 0, pixelCount * sizeof(glm::vec3));

    profiler.generate.recordStart();
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, maxTraceDepth, enableDepthOfField, dev_paths);
    profiler.generate.recordStop();
    checkCUDAError("generate camera ray");

    int activePathCount = pixelCount;
    int tracedDepth = 0;

    for (int depth = 0; depth < maxTraceDepth && activePathCount > 0; depth++)
    {
        dim3 numblocksPathSegmentTracing = (activePathCount + blockSize1d - 1) / blockSize1d;

        profiler.intersections[depth].recordStart();
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            activePathCount,
            dev_paths,
            dev_geoms,
            dev_triangles,
            dev_primitives,
            static_cast<int>(hst_scene->primitives.size()),
            dev_bvhNodes,
            static_cast<int>(hst_scene->bvhNodes.size()),
            useBvh,
            materialCount,
            dev_intersections,
            dev_materialKeys
        );
        profiler.intersections[depth].recordStop();
        checkCUDAError("compute intersections");

        if (sortByMaterial)
        {
            profiler.sorts[depth].recordStart();

            if (activePathCount > 1)
            {
                auto keyBegin = thrust::device_pointer_cast(dev_materialKeys);
                auto pathBegin = thrust::device_pointer_cast(dev_paths);
                auto intersectionBegin = thrust::device_pointer_cast(dev_intersections);
                auto valuesBegin = thrust::make_zip_iterator(thrust::make_tuple(pathBegin, intersectionBegin));

                thrust::sort_by_key(
                    thrust::device,
                    keyBegin,
                    keyBegin + activePathCount,
                    valuesBegin);  // sorts both paths and intersections together
            }

            profiler.sorts[depth].recordStop();
            checkCUDAError("sort paths by material");
        }

        profiler.shades[depth].recordStart();
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            activePathCount,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_environmentTexture,
            enableEnvironment,
            hst_scene->environment.intensity,
            hst_scene->environment.rotation,
            dev_iterationImage);
        profiler.shades[depth].recordStop();
        checkCUDAError("shade");

        if (enableCompaction)
        {
            profiler.compactions[depth].recordStart();
            auto pathBegin = thrust::device_pointer_cast(dev_paths);
            const auto newEnd = thrust::remove_if(
                thrust::device,
                pathBegin,
                pathBegin + activePathCount,
                IsPathTerminated{});
            activePathCount = static_cast<int>(newEnd - pathBegin);
            profiler.compactions[depth].recordStop();
            checkCUDAError("compact terminated paths");
        }

        tracedDepth = depth + 1;
        if (guiData != nullptr)
        {
            guiData->TracedDepth = tracedDepth;
            guiData->ActivePathsByDepth.push_back(activePathCount);
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelCount + blockSize1d - 1) / blockSize1d;
    profiler.gather.recordStart();
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelCount, dev_image, dev_iterationImage);
    profiler.gather.recordStop();
    checkCUDAError("final gather");

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    profiler.writeResults(guiData, tracedDepth, sortByMaterial, enableCompaction);

    checkCUDAError("pathtrace");
}
