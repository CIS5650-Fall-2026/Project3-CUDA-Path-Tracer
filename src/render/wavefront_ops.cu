#include "render/wavefront_ops.h"

#include "utilities.h"

#include <cub/device/device_radix_sort.cuh>

#include <type_traits>
#include <utility>

namespace
{
int capacity = 0;
PathSegment* dev_sparePaths = nullptr;
ShadeableIntersection* dev_spareIntersections = nullptr;
int* dev_indices = nullptr;        // 0, 1, 2, ... as the sort's values
int* dev_sortedIndices = nullptr;  // source index of each sorted position
int* dev_sortedKeys = nullptr;     // required by CUB, unused afterwards
void* dev_cubTemp = nullptr;
size_t cubTempBytes = 0;

// Slots move on per bounce and carry over between iterations, so no reset is needed
int* dev_survivorCounts = nullptr;
int survivorSlot = 0;

ShadowRay* dev_shadowRays = nullptr;
int* dev_shadowCounts = nullptr;
int shadowSlot = 0;

// Also catches a missing wavefrontInit: CUB treats a null temp pointer as a size query
void checkCapacity(int numPaths)
{
    if (numPaths > capacity)
    {
        fatal("wavefront_ops: %d paths, the workspace holds %d", numPaths, capacity);
    }
}

__global__ void fillIndices(int n, int* indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        indices[i] = i;
    }
}

// The struct copies must be memcpys (all loads, then all stores): a
// hand-written glm operator= once made them member-wise and this kernel 3 to 4x slower.
static_assert(std::is_trivially_copyable<PathSegment>::value,
    "PathSegment must be trivially copyable so the gather copies it as a block");
static_assert(std::is_trivially_copyable<ShadeableIntersection>::value,
    "ShadeableIntersection must be trivially copyable so the gather copies it as a block");
__global__ void gatherByIndex(int n, const int* sortedIndices,
    const PathSegment* pathsIn, const ShadeableIntersection* intersectionsIn,
    PathSegment* pathsOut, ShadeableIntersection* intersectionsOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int src = sortedIndices[i];
        pathsOut[i] = pathsIn[src];
        intersectionsOut[i] = intersectionsIn[src];
    }
}
}  // namespace

void wavefrontInit(int maxPaths)
{
    capacity = maxPaths;
    CUDA_CHECK(cudaMalloc(&dev_sparePaths, maxPaths * sizeof(PathSegment)));
    CUDA_CHECK(cudaMalloc(&dev_spareIntersections, maxPaths * sizeof(ShadeableIntersection)));
    CUDA_CHECK(cudaMalloc(&dev_indices, maxPaths * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_sortedIndices, maxPaths * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_sortedKeys, maxPaths * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_survivorCounts, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemset(dev_survivorCounts, 0, 3 * sizeof(int)));
    survivorSlot = 0;
    CUDA_CHECK(cudaMalloc(&dev_shadowRays, maxPaths * sizeof(ShadowRay)));
    CUDA_CHECK(cudaMalloc(&dev_shadowCounts, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(dev_shadowCounts, 0, 2 * sizeof(int)));
    shadowSlot = 0;

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, cubTempBytes, dev_sortedKeys, dev_sortedKeys,
        dev_indices, dev_sortedIndices, maxPaths));
    CUDA_CHECK(cudaMalloc(&dev_cubTemp, cubTempBytes));
}

void wavefrontFree()
{
    cudaFree(dev_sparePaths);
    cudaFree(dev_spareIntersections);
    cudaFree(dev_indices);
    cudaFree(dev_sortedIndices);
    cudaFree(dev_sortedKeys);
    cudaFree(dev_survivorCounts);
    cudaFree(dev_shadowRays);
    cudaFree(dev_shadowCounts);
    cudaFree(dev_cubTemp);

    dev_sparePaths = nullptr;
    dev_spareIntersections = nullptr;
    dev_indices = nullptr;
    dev_sortedIndices = nullptr;
    dev_sortedKeys = nullptr;
    dev_survivorCounts = nullptr;
    survivorSlot = 0;
    dev_shadowRays = nullptr;
    dev_shadowCounts = nullptr;
    shadowSlot = 0;
    dev_cubTemp = nullptr;
    cubTempBytes = 0;
    capacity = 0;
}

SurvivorBuffer wavefrontSurvivors()
{
    return { dev_sparePaths, dev_survivorCounts + survivorSlot, dev_survivorCounts + (survivorSlot + 1) % 3 };
}

SurvivorBuffer wavefrontSurvivorsInPlace(PathSegment* paths)
{
    return { paths, dev_survivorCounts + survivorSlot, dev_survivorCounts + (survivorSlot + 1) % 3 };
}

ShadowQueue wavefrontShadowQueue()
{
    return { dev_shadowRays, dev_shadowCounts + shadowSlot, dev_shadowCounts + (shadowSlot ^ 1) };
}

ShadowQueue wavefrontSwapShadowQueue()
{
    const ShadowQueue filled = { dev_shadowRays, dev_shadowCounts + shadowSlot, nullptr };
    shadowSlot ^= 1;
    return filled;
}

int wavefrontCountSurvivors(cudaStream_t stream)
{
    // The one synchronization per bounce
    int numAlive = 0;
    CUDA_CHECK(cudaMemcpyAsync(&numAlive, dev_survivorCounts + survivorSlot, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    survivorSlot = (survivorSlot + 1) % 3;
    return numAlive;
}

int wavefrontSwapPaths(PathSegment*& paths, cudaStream_t stream)
{
    const int numAlive = wavefrontCountSurvivors(stream);
    std::swap(paths, dev_sparePaths);
    return numAlive;
}

void sortMaterials(int numPaths, int keyBits, const int* materialIds,
                   ShadeableIntersection*& intersections, PathSegment*& paths,
                   cudaStream_t stream)
{
    checkCapacity(numPaths);

    const int numBlocks = (numPaths + PATH_BLOCK_SIZE - 1) / PATH_BLOCK_SIZE;
    fillIndices<<<numBlocks, PATH_BLOCK_SIZE, 0, stream>>>(numPaths, dev_indices);

    // CUB sorts 8 bits per pass, so only the low keyBits bits take part
    size_t bytes = cubTempBytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dev_cubTemp, bytes,
        materialIds, dev_sortedKeys, dev_indices, dev_sortedIndices,
        numPaths, 0, keyBits, stream));

    gatherByIndex<<<numBlocks, PATH_BLOCK_SIZE, 0, stream>>>(numPaths, dev_sortedIndices,
        paths, intersections, dev_sparePaths, dev_spareIntersections);

    std::swap(paths, dev_sparePaths);
    std::swap(intersections, dev_spareIntersections);
}
