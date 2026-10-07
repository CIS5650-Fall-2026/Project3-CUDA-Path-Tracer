#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <stdexcept>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    const cudaError_t syncError = cudaDeviceSynchronize();
    const cudaError_t launchError = cudaGetLastError();
    const cudaError_t err = syncError != cudaSuccess ? syncError : launchError;
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

static void requireCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct IsActive {
    __host__ __device__
        bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

struct CompareMaterial {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a,
            const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    // Hash each dimension without signed shifts or overlapping bit fields.
    unsigned int seed = utilhash(static_cast<unsigned int>(iter));
    seed = utilhash(seed ^ static_cast<unsigned int>(index));
    seed = utilhash(seed ^ static_cast<unsigned int>(depth));
    return thrust::default_random_engine(seed == 0u ? 1u : seed);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = iter > 0 ? image[index] : glm::vec3(0.0f);
        const int samples = iter > 0 ? iter : 1;

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / samples * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / samples * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / samples * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene, bool restoreAccumulation)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    if (scene->state.image.size() != static_cast<std::size_t>(pixelcount)) {
        throw std::runtime_error("Host accumulation size does not match the camera");
    }
    requireCuda(cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3)), "allocate image");
    if (restoreAccumulation) {
        requireCuda(cudaMemcpy(dev_image, scene->state.image.data(),
            pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice), "restore image");
    }
    else {
        std::fill(scene->state.image.begin(), scene->state.image.end(), glm::vec3(0.0f));
        requireCuda(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)), "clear image");
    }

    requireCuda(cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment)), "allocate paths");

    if (!scene->geoms.empty()) {
        requireCuda(cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom)), "allocate geometry");
        requireCuda(cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
            cudaMemcpyHostToDevice), "upload geometry");
    }

    if (!scene->materials.empty()) {
        requireCuda(cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material)), "allocate materials");
        requireCuda(cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
            cudaMemcpyHostToDevice), "upload materials");
    }

    requireCuda(cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection)), "allocate hits");


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    dev_image = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_materials = nullptr;
    dev_intersections = nullptr;

    checkCUDAError("pathtraceFree");
}

void pathtraceDisplay(uchar4* pbo, int completedSamples)
{
    const Camera& cam = hst_scene->state.camera;
    const dim3 block(8, 8);
    const dim3 grid((cam.resolution.x + 7) / 8, (cam.resolution.y + 7) / 8);
    sendImageToPBO<<<grid, block>>>(pbo, cam.resolution, completedSamples, dev_image);
    checkCUDAError("display restored image");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
    bool antialiasing, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = traceDepth > 0 ? glm::vec3(1.0f) : glm::vec3(0.0f);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float sampleX = static_cast<float>(x) + (antialiasing ? u01(rng) : 0.5f);
        const float sampleY = static_cast<float>(y) + (antialiasing ? u01(rng) : 0.5f);

        // One uniform subpixel sample per pixel per iteration.
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// Every active path gets a complete hit record, including initialized misses.
__global__ void computeIntersections(
    int depth, int num_paths, PathSegment* pathSegments,
    Geom* geoms, int geoms_size, ShadeableIntersection* intersections)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    ShadeableIntersection best{-1.0f, glm::vec3(0.0f), -1, glm::vec3(0.0f)};
    if (pathSegments[idx].remainingBounces <= 0) {
        // Sorting must receive a defined key even for completed paths.
        intersections[idx] = best;
        return;
    }
    const Ray ray = pathSegments[idx].ray;
    float closest = FLT_MAX;
    for (int i = 0; i < geoms_size; ++i) {
        glm::vec3 position(0.0f), normal(0.0f);
        bool outside = true;
        float t = -1.0f;
        if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], ray, position, normal, outside);
        }
        else if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], ray, position, normal, outside);
        }
        if (t > 0.0f && t < closest) {
            closest = t;
            best.t = t;
            best.materialId = geoms[i].materialid;
            best.surfaceNormal = normal; // Still outward, never face-forwarded.
            best.position = position;    // Preserve the exact intersection output.
        }
    }
    intersections[idx] = best;
}

__global__ void shadeMaterial(
    int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments, Material* materials)
{
    int idx;
    if (!utilityCore::indexIsValid(num_paths, idx)) return;
    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0) return;
    const ShadeableIntersection& hit = shadeableIntersections[idx];
    if (hit.t <= 0.0f) {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return;
    }

    const Material& material = materials[hit.materialId];
    const glm::vec3 emitted = material.emission.color * material.emission.emittance;
    if (emitted.x > 0.0f || emitted.y > 0.0f || emitted.z > 0.0f) {
        path.color *= emitted;
        path.remainingBounces = 0;
        return;
    }

   
    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, path.pixelIndex, path.remainingBounces);
    scatterRay(path, hit.position, hit.surfaceNormal, material, rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter,
    std::vector<PathTraceBounce>* bounceCounts)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, hst_scene->state.antialiasing, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = traceDepth > 0 ? pixelcount : 0;
    if (bounceCounts) {
        bounceCounts->clear();
        bounceCounts->reserve(traceDepth + 1);
        bounceCounts->push_back({0, num_paths, 0});
    }
    if (guiData != nullptr) guiData->TracedDepth = 0;

    // Without compaction, launch the whole buffer for a fixed depth budget.
    // Dead paths keep their final contribution and are skipped by both kernels.
    while (num_paths > 0 && depth < traceDepth) {
        const int launchedPaths = num_paths;
        const int blocks = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<blocks, blockSize1d>>>(
            depth, num_paths, dev_paths, dev_geoms,
            static_cast<int>(hst_scene->geoms.size()), dev_intersections);
        checkCUDAError("trace one bounce");
        ++depth;

        if (hst_scene->state.sortMaterials) {
            // Sort hit keys and corresponding paths together, only in the
            // active prefix. Material-array indices unchanged.
            thrust::sort_by_key(thrust::device,
                dev_intersections, dev_intersections + num_paths,
                dev_paths, CompareMaterial{});
        }

        shadeMaterial<<<blocks, blockSize1d>>>(
            iter, num_paths, dev_intersections, dev_paths, dev_materials);
        checkCUDAError("shade one bounce");

        
        if (hst_scene->state.compactPaths) {
            PathSegment* activeEnd = thrust::partition(thrust::device,
                dev_paths, dev_paths + num_paths, IsActive{});
            num_paths = static_cast<int>(activeEnd - dev_paths);
        }
        if (bounceCounts) {
            bounceCounts->push_back({depth,
                hst_scene->state.compactPaths ? num_paths : -1, launchedPaths});
        }
        if (guiData != nullptr) guiData->TracedDepth = depth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    finalGather<<<numBlocksPixels, blockSize1d>>>(
        pixelcount,
        dev_image,
        dev_paths
        );

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    requireCuda(cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost), "read completed image");

    checkCUDAError("pathtrace");
}
