#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cub/cub.cuh>

// 0 = off, 1 = thrust::sort_by_key on payloads, 2 = CUB pairs + gather
#define SORT_BY_MATERIAL 0

// 0 = off, 1 = thrust::partition, 2 = work-efficient scan (Project 2)
#define COMPACTION_MODE 2

// prints surviving path counts per bounce
#define PERF_ANALYSIS 0

#define RUSSIAN_ROULETTE 1
#define RR_START_BOUNCE 3

#define USE_BVH 1

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "stream_compaction/efficient.h"

// every check is a full device sync, so keep it off unless debugging
#define ERRORCHECK 0

#undef checkCUDAError
#if ERRORCHECK
#define checkCUDAError(msg) do { cudaDeviceSynchronize(); checkCUDAErrorFn(msg, FILENAME, __LINE__); } while (0)
#else
#define checkCUDAError(msg) do {} while (0)
#endif

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

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialKeys = NULL;
static PathSegment* dev_pathsScratch = NULL;
static int* dev_aliveFlags = NULL;
static int* dev_scanIndices = NULL;
static int* dev_aliveCount = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhNodes = NULL;
static int* dev_materialKeysOut = NULL;
static int* dev_sortIndices = NULL;
static int* dev_sortIndicesOut = NULL;
static ShadeableIntersection* dev_intersectionsScratch = NULL;
static void* dev_sortTemp = NULL;
static size_t dev_sortTempBytes = 0;

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

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));

    // allocate once. a cudaMalloc per bounce costs more than the compaction.
    // scan needs a power of two, hence paddedCount.
    const int paddedCount = 1 << ilog2ceil(pixelcount);
    cudaMalloc(&dev_pathsScratch, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_aliveFlags, pixelcount * sizeof(int));
    cudaMalloc(&dev_scanIndices, paddedCount * sizeof(int));
    cudaMalloc(&dev_aliveCount, sizeof(int));

#if SORT_BY_MATERIAL == 2
    cudaMalloc(&dev_materialKeysOut, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_sortIndicesOut, pixelcount * sizeof(int));
    cudaMalloc(&dev_intersectionsScratch, pixelcount * sizeof(ShadeableIntersection));

    // temp size only grows with n, so size it for the worst case
    cub::DeviceRadixSort::SortPairs(nullptr, dev_sortTempBytes,
        dev_materialKeys, dev_materialKeysOut,
        dev_sortIndices, dev_sortIndicesOut, pixelcount);
    cudaMalloc(&dev_sortTemp, dev_sortTempBytes);
#endif

    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(),
            scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(),
            scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_materialKeys);
    cudaFree(dev_pathsScratch);
    cudaFree(dev_aliveFlags);
    cudaFree(dev_scanIndices);
    cudaFree(dev_aliveCount);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_materialKeysOut);
    cudaFree(dev_sortIndices);
    cudaFree(dev_sortIndicesOut);
    cudaFree(dev_intersectionsScratch);
    cudaFree(dev_sortTemp);
    dev_triangles = NULL;
    dev_bvhNodes = NULL;
    dev_sortTemp = NULL;
    dev_sortTempBytes = 0;

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // AA: jitter inside the pixel, new offset every iteration
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (((float)x + jitterX) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (((float)y + jitterY) - (float)cam.resolution.y * 0.5f)
        );

        if (cam.lensRadius > 0.0f)
        {
            // thin lens: jitter the origin over the aperture, then re-aim
            // at the focal plane
            float r = cam.lensRadius * sqrtf(u01(rng));
            float theta = u01(rng) * TWO_PI;
            glm::vec3 lensOffset = cam.right * (r * cosf(theta)) + cam.up * (r * sinf(theta));

            float focusT = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 focusPoint = segment.ray.origin + segment.ray.direction * focusT;

            segment.ray.origin += lensOffset;
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    const Triangle* triangles,
    int triangles_size,
    const BVHNode* bvhNodes,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        int hit_material = -1;
        bool outside = true;
        bool hit_outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                hit_outside = outside;
                hit_material = geoms[i].materialid;
            }
        }

        // t_min doubles as the traversal cutoff
        if (triangles_size > 0)
        {
            glm::vec3 tri_intersect;
            glm::vec3 tri_normal;
            bool tri_outside = true;
            int tri_material = -1;

#if USE_BVH
            float t_tri = bvhIntersectionTest(bvhNodes, triangles, pathSegment.ray, t_min,
                tri_intersect, tri_normal, tri_outside, tri_material);
#else
            float t_tri = meshBruteForceIntersectionTest(triangles, triangles_size, pathSegment.ray,
                t_min, tri_intersect, tri_normal, tri_outside, tri_material);
#endif

            if (t_tri > 0.0f && t_tri < t_min)
            {
                t_min = t_tri;
                normal = tri_normal;
                hit_outside = tri_outside;
                hit_material = tri_material;
            }
        }

        if (hit_material == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hit_material;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = hit_outside;
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0)
    {
        // only reachable with compaction off
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t <= 0.0f)
    {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];

    if (material.emittance > 0.0f)
    {
        path.color *= material.color * material.emittance;
        path.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
    glm::vec3 intersectPoint = getPointOnRay(path.ray, intersection.t);
    scatterRay(path, intersectPoint, intersection.surfaceNormal, intersection.outside, material, rng);

    if (path.remainingBounces <= 0)
    {
        // ran out of bounces without reaching a light
        path.color = glm::vec3(0.0f);
        return;
    }

#if RUSSIAN_ROULETTE
    if (depth >= RR_START_BOUNCE)
    {
        // q tracks throughput. the 1/q boost keeps it unbiased.
        thrust::uniform_real_distribution<float> u01(0, 1);
        float q = fminf(0.95f, fmaxf(path.color.x, fmaxf(path.color.y, path.color.z)));
        if (u01(rng) >= q)
        {
            path.color = glm::vec3(0.0f);
            path.remainingBounces = 0;
        }
        else
        {
            path.color /= q;
        }
    }
#endif
}

__global__ void kernExtractMaterialKeys(
    int num_paths,
    int* keys,
    const ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        keys[idx] = intersections[idx].materialId;
    }
}

__global__ void kernExtractMaterialKeysAndIndices(
    int num_paths,
    int* keys,
    int* indices,
    const ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        keys[idx] = intersections[idx].materialId;
        indices[idx] = idx;
    }
}

// gather rather than scatter, so the writes stay coalesced
__global__ void kernGatherByIndex(
    int num_paths,
    const int* indices,
    const PathSegment* pathsIn,
    const ShadeableIntersection* intersectionsIn,
    PathSegment* pathsOut,
    ShadeableIntersection* intersectionsOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        int src = indices[idx];
        pathsOut[idx] = pathsIn[src];
        intersectionsOut[idx] = intersectionsIn[src];
    }
}

struct PathIsAlive
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces > 0;
    }
};

__global__ void kernMapAliveFlags(int num_paths, int* flags, const PathSegment* paths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        flags[idx] = paths[idx].remainingBounces > 0 ? 1 : 0;
    }
}

// dead paths get written too - finalGather still needs their color
__global__ void kernPartitionPaths(
    int num_paths,
    const int* numAlive,
    PathSegment* dst,
    const PathSegment* src,
    const int* flags,
    const int* scanIndices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    int aliveBefore = scanIndices[idx];
    int dstIdx = flags[idx] ? aliveBefore : *numAlive + (idx - aliveBefore);
    dst[dstIdx] = src[idx];
}

// total = last scan entry + last flag
__global__ void kernComputeAliveCount(int num_paths, const int* flags, const int* scanIndices, int* out)
{
    *out = scanIndices[num_paths - 1] + flags[num_paths - 1];
}

static int compactPathsCustom(int num_paths, int blockSize)
{
    if (num_paths <= 0)
    {
        return 0;
    }

    dim3 blocks((num_paths + blockSize - 1) / blockSize);
    kernMapAliveFlags<<<blocks, blockSize>>>(num_paths, dev_aliveFlags, dev_paths);

    const int paddedN = 1 << ilog2ceil(num_paths);
    cudaMemset(dev_scanIndices, 0, paddedN * sizeof(int));
    cudaMemcpy(dev_scanIndices, dev_aliveFlags, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
    StreamCompaction::Efficient::devScan(paddedN, dev_scanIndices);

    kernComputeAliveCount<<<1, 1>>>(num_paths, dev_aliveFlags, dev_scanIndices, dev_aliveCount);

    // the partition reads the count on device; the host needs it for the next launch size
    kernPartitionPaths<<<blocks, blockSize>>>(
        num_paths, dev_aliveCount, dev_pathsScratch, dev_paths, dev_aliveFlags, dev_scanIndices);

    // copy back into the same slots, so the dead stay parked
    cudaMemcpy(dev_paths, dev_pathsScratch, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

    int numAlive = 0;
    cudaMemcpy(&numAlive, dev_aliveCount, sizeof(int), cudaMemcpyDeviceToHost);
    return numAlive;
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
void pathtrace(uchar4* pbo, int frame, int iter)
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

    ///////////////////////////////////////////////////////////////////////////

    // per depth: intersect -> (sort) -> shade -> compact
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            (int)hst_scene->triangles.size(),
            dev_bvhNodes,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        depth++;

        // group by material to cut divergence in shadeMaterial
#if SORT_BY_MATERIAL == 1
        kernExtractMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, dev_materialKeys, dev_intersections);
        {
            // permutes 68B per element, and allocates every call
            thrust::device_ptr<int> keysBegin(dev_materialKeys);
            auto valuesBegin = thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_ptr<PathSegment>(dev_paths),
                thrust::device_ptr<ShadeableIntersection>(dev_intersections)));
            thrust::sort_by_key(keysBegin, keysBegin + num_paths, valuesBegin);
        }
        checkCUDAError("sort by material");
#elif SORT_BY_MATERIAL == 2
        // sort 8B keys+indices instead of the payload, then gather once
        kernExtractMaterialKeysAndIndices<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, dev_materialKeys, dev_sortIndices, dev_intersections);

        {
            size_t tempBytes = dev_sortTempBytes;
            cub::DeviceRadixSort::SortPairs(dev_sortTemp, tempBytes,
                dev_materialKeys, dev_materialKeysOut,
                dev_sortIndices, dev_sortIndicesOut, num_paths);
        }

        kernGatherByIndex<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, dev_sortIndicesOut, dev_paths, dev_intersections,
            dev_pathsScratch, dev_intersectionsScratch);

        // copy back instead of swapping - dev_paths also holds the dead
        // paths parked past num_paths
        cudaMemcpy(dev_paths, dev_pathsScratch,
            num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_intersections, dev_intersectionsScratch,
            num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        checkCUDAError("sort by material");
#endif

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        checkCUDAError("shade material");

#if PERF_ANALYSIS
        const int launchedThisBounce = num_paths;
#endif

        // alive to the front. this round's dead land just past the new
        // boundary, which later bounces never reach.
#if COMPACTION_MODE == 1
        {
            thrust::device_ptr<PathSegment> pathsBegin(dev_paths);
            auto newEnd = thrust::partition(pathsBegin, pathsBegin + num_paths, PathIsAlive());
            num_paths = static_cast<int>(newEnd - pathsBegin);
        }
#elif COMPACTION_MODE == 2
        num_paths = compactPathsCustom(num_paths, blockSize1d);
#endif

#if PERF_ANALYSIS
        {
            thrust::device_ptr<PathSegment> countBegin(dev_paths);
            int alive = static_cast<int>(
                thrust::count_if(countBegin, countBegin + launchedThisBounce, PathIsAlive()));
            printf("iter %d depth %d: %d launched, %d survived\n",
                iter, depth, launchedThisBounce, alive);
        }
#endif

        iterationComplete = (num_paths == 0) || (depth >= traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // pixelcount, not num_paths. the dead are still in there, only moved,
    // and each carries its own pixelIndex.
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    checkCUDAError("pathtrace");
}

// only on save. this used to run every iteration and nothing read it.
void pathtraceCopyImageToHost()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("copy image to host");
}
