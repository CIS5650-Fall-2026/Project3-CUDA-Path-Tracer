#include "render/exit_pupil.h"

#include "utilities.h"

#include <cuda_runtime.h>

#include <cmath>

namespace
{
constexpr int PAD_CELLS = 2;

// Cell i of the grid over the rear element's square, paired with a film point spread over the bin's radii.
__host__ __device__ inline bool cellPasses(const LensDevice& lens, int grid, int i, float r0, float r1, int& ix, int& iy)
{
    ix = i % grid;
    iy = i / grid;
    const float cell = 2.0f * lens.aimRadius / (float)grid;
    const float x = -lens.aimRadius + ((float)ix + 0.5f) * cell;
    const float y = -lens.aimRadius + ((float)iy + 0.5f) * cell;
    const float t = ((float)i + 0.5f) * 0.6180339887f;
    const float filmX = r0 + (r1 - r0) * (t - floorf(t));
    glm::vec3 o(filmX, 0.0f, lens.filmDistance);
    glm::vec3 d = glm::normalize(glm::vec3(x, y, lens.rearZ) - o);
    return traceLens(lens.surfaces, lens.count - 1, 0, lens.rearZ, o, d);
}

__host__ __device__ inline glm::vec4 cellsToBox(const LensDevice& lens, int grid, int minX, int maxX, int minY, int maxY)
{
    if (maxX < minX)
    {
        return glm::vec4(0.0f);
    }
    const float cell = 2.0f * lens.aimRadius / (float)grid;
    const float R = lens.aimRadius;
    return glm::vec4(
        -R + (float)(minX - PAD_CELLS) * cell,
        -R + (float)(minY - PAD_CELLS) * cell,
        -R + (float)(maxX + 1 + PAD_CELLS) * cell,
        -R + (float)(maxY + 1 + PAD_CELLS) * cell);
}

__host__ __device__ inline void binRadii(const LensDevice& lens, int bin, float& r0, float& r1)
{
    r0 = lens.halfDiagonal * (float)bin / (float)lens.pupilBinCount;
    r1 = lens.halfDiagonal * (float)(bin + 1) / (float)lens.pupilBinCount;
}

__host__ __device__ inline float cellArea(const LensDevice& lens, int grid)
{
    const float cell = 2.0f * lens.aimRadius / (float)grid;
    return cell * cell;
}

__global__ void buildBinKernel(LensDevice lens, int grid, glm::vec4* boxes, float* areas)
{
    __shared__ int sMinX, sMaxX, sMinY, sMaxY, sCount;
    if (threadIdx.x == 0)
    {
        sMinX = grid;
        sMaxX = -1;
        sMinY = grid;
        sMaxY = -1;
        sCount = 0;
    }
    __syncthreads();
    float r0;
    float r1;
    binRadii(lens, blockIdx.x, r0, r1);
    for (int i = threadIdx.x; i < grid * grid; i += blockDim.x)
    {
        int ix;
        int iy;
        if (cellPasses(lens, grid, i, r0, r1, ix, iy))
        {
            atomicMin(&sMinX, ix);
            atomicMax(&sMaxX, ix);
            atomicMin(&sMinY, iy);
            atomicMax(&sMaxY, iy);
            atomicAdd(&sCount, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        boxes[blockIdx.x] = cellsToBox(lens, grid, sMinX, sMaxX, sMinY, sMaxY);
        areas[blockIdx.x] = (float)sCount * cellArea(lens, grid);
    }
}
}  // namespace

int exitPupilBinCount(float halfDiagonal)
{
    return (int)ceilf(halfDiagonal / EXIT_PUPIL_BIN_MM);
}

void exitPupilBuildHost(const LensDevice& lens, int grid, std::vector<glm::vec4>& boxes, std::vector<float>& areas)
{
    boxes.assign(lens.pupilBinCount, glm::vec4(0.0f));
    areas.assign(lens.pupilBinCount, 0.0f);
    for (int bin = 0; bin < lens.pupilBinCount; ++bin)
    {
        float r0;
        float r1;
        binRadii(lens, bin, r0, r1);
        int minX = grid;
        int maxX = -1;
        int minY = grid;
        int maxY = -1;
        int count = 0;
        for (int i = 0; i < grid * grid; ++i)
        {
            int ix;
            int iy;
            if (cellPasses(lens, grid, i, r0, r1, ix, iy))
            {
                minX = ix < minX ? ix : minX;
                maxX = ix > maxX ? ix : maxX;
                minY = iy < minY ? iy : minY;
                maxY = iy > maxY ? iy : maxY;
                ++count;
            }
        }
        boxes[bin] = cellsToBox(lens, grid, minX, maxX, minY, maxY);
        areas[bin] = (float)count * cellArea(lens, grid);
    }
}

float exitPupilBuildDevice(const LensDevice& lens, int grid, glm::vec4* boxes, float* areas)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    buildBinKernel<<<lens.pupilBinCount, 256>>>(lens, grid, boxes, areas);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}
