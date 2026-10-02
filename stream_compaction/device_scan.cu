#include "device_scan.cuh"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

// This is the renderer-local extraction of our own CIS 565 Project 2 scan,
// revision 6a13e6fbe3a998c67fdbbdbe6643a5b4c4233b31, originally authored in
// stream_compaction/shared.cu and shared_internal.cuh.  It intentionally omits
// Project 2's host wrappers, common.h, and PerformanceTimer dependency.
namespace cis565 {
namespace scan {
namespace {

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
            + cudaGetErrorString(result));
    }
}

std::size_t ceilDiv(std::size_t n, std::size_t divisor) {
    return n / divisor + (n % divisor != 0);
}

__host__ __device__ std::size_t paddedIndex(std::size_t index,
        bool padding) {
    return padding ? index + (index >> 5U) + (index >> 10U) : index;
}

__global__ void scanEfficientTiles(std::size_t n, int* output,
        const int* input, int* tileTotals, bool bankPadding) {
    extern __shared__ int scratch[];
    const std::size_t lane = threadIdx.x;
    const std::size_t half = blockDim.x;
    const std::size_t tileElements = 2U * half;
    const std::size_t tileStart = static_cast<std::size_t>(blockIdx.x)
        * tileElements;
    const std::size_t first = tileStart + lane;
    const std::size_t second = first + half;

    // The last global input does not contribute to an exclusive result.
    scratch[paddedIndex(lane, bankPadding)] =
        first < n && first + 1U < n ? input[first] : 0;
    scratch[paddedIndex(lane + half, bankPadding)] =
        second < n && second + 1U < n ? input[second] : 0;
    __syncthreads();

    for (std::size_t offset = 1; offset < tileElements; offset <<= 1U) {
        const std::size_t right = (lane + 1U) * (offset << 1U) - 1U;
        if (right < tileElements) {
            const std::size_t left = right - offset;
            scratch[paddedIndex(right, bankPadding)] +=
                scratch[paddedIndex(left, bankPadding)];
        }
        __syncthreads();
    }

    const std::size_t root = tileElements - 1U;
    if (lane == 0) {
        if (tileTotals != nullptr) {
            tileTotals[blockIdx.x] = scratch[paddedIndex(root, bankPadding)];
        }
        scratch[paddedIndex(root, bankPadding)] = 0;
    }
    __syncthreads();

    for (std::size_t offset = tileElements >> 1U; offset > 0;
            offset >>= 1U) {
        const std::size_t right = (lane + 1U) * (offset << 1U) - 1U;
        if (right < tileElements) {
            const std::size_t left = right - offset;
            const std::size_t paddedLeft = paddedIndex(left, bankPadding);
            const std::size_t paddedRight = paddedIndex(right, bankPadding);
            const int leftValue = scratch[paddedLeft];
            const int rightValue = scratch[paddedRight];
            scratch[paddedLeft] = rightValue;
            scratch[paddedRight] = rightValue + leftValue;
        }
        __syncthreads();
    }

    if (first < n) {
        output[first] = scratch[paddedIndex(lane, bankPadding)];
    }
    if (second < n) {
        output[second] = scratch[paddedIndex(lane + half, bankPadding)];
    }
}

__global__ void addTileOffsets(std::size_t n, int* values,
        const int* tileOffsets) {
    const std::size_t lane = threadIdx.x;
    const std::size_t tileElements = 2U * static_cast<std::size_t>(blockDim.x);
    const std::size_t first = static_cast<std::size_t>(blockIdx.x)
        * tileElements + lane;
    const int offset = tileOffsets[blockIdx.x];
    if (first < n) {
        values[first] += offset;
    }
    const std::size_t second = first + blockDim.x;
    if (second < n) {
        values[second] += offset;
    }
}

bool sameConfig(const Config& a, const Config& b) {
    return a.algorithm == b.algorithm && a.blockSize == b.blockSize
        && a.bankPadding == b.bankPadding;
}

void launchLevel(std::size_t n, int* output, const int* input,
        const Config& config, ScanWorkspace& workspace, std::size_t level) {
    const std::size_t blockCount = ceilDiv(n, workspace.tileSize());
    int* totals = blockCount > 1 ? workspace.levelTotals(level) : nullptr;
    const dim3 blocks(static_cast<unsigned int>(blockCount));
    const dim3 threads(static_cast<unsigned int>(config.blockSize));
    const std::size_t physicalElements = paddedIndex(
        workspace.tileSize() - 1U, config.bankPadding) + 1U;

    scanEfficientTiles<<<blocks, threads, physicalElements * sizeof(int)>>>(
        n, output, input, totals, config.bankPadding);
    cudaCheck(cudaGetLastError(), "launch exclusive scan");
    if (blockCount == 1) {
        return;
    }

    int* offsets = workspace.levelOffsets(level);
    launchLevel(blockCount, offsets, totals, config, workspace, level + 1U);
    addTileOffsets<<<blocks, threads>>>(n, output, offsets);
    cudaCheck(cudaGetLastError(), "launch scan uniform add");
}

} // namespace

ScanWorkspace::ScanWorkspace(std::size_t capacity, Config config)
    : capacity_(capacity), tileSize_(0), config_(config) {
    if (config.algorithm != Algorithm::Efficient) {
        throw std::invalid_argument("only the efficient scan is available");
    }
    if (config.blockSize < 32 || config.blockSize > 1024
            || (config.blockSize & (config.blockSize - 1)) != 0) {
        throw std::invalid_argument(
            "scan block size must be a power of two from 32 through 1024");
    }
    tileSize_ = 2U * static_cast<std::size_t>(config.blockSize);
    std::size_t count = capacity_;
    try {
        while (count > tileSize_) {
            const std::size_t blocks = ceilDiv(count, tileSize_);
            if (blocks > std::numeric_limits<std::size_t>::max() / sizeof(int)) {
                throw std::overflow_error("scan workspace allocation overflow");
            }
            int* totals = nullptr;
            int* offsets = nullptr;
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&totals),
                blocks * sizeof(int)), "allocate scan totals");
            try {
                cudaCheck(cudaMalloc(reinterpret_cast<void**>(&offsets),
                    blocks * sizeof(int)), "allocate scan offsets");
            } catch (...) {
                cudaFree(totals);
                throw;
            }
            totals_.push_back(totals);
            offsets_.push_back(offsets);
            count = blocks;
        }
    } catch (...) {
        for (int* pointer : totals_) {
            cudaFree(pointer);
        }
        for (int* pointer : offsets_) {
            cudaFree(pointer);
        }
        totals_.clear();
        offsets_.clear();
        throw;
    }
}

ScanWorkspace::~ScanWorkspace() {
    for (int* pointer : totals_) {
        cudaFree(pointer);
    }
    for (int* pointer : offsets_) {
        cudaFree(pointer);
    }
}

void scanDevice(std::size_t n, int* deviceOutput, const int* deviceInput,
        const Config& config, ScanWorkspace& workspace) {
    if (n == 0) {
        return;
    }
    if (deviceOutput == nullptr || deviceInput == nullptr) {
        throw std::invalid_argument("nonempty device scan needs valid pointers");
    }
    if (deviceOutput == deviceInput) {
        throw std::invalid_argument("device scan input and output must differ");
    }
    if (n > workspace.capacity() || !sameConfig(config, workspace.config())) {
        throw std::invalid_argument("device scan workspace does not match request");
    }
    launchLevel(n, deviceOutput, deviceInput, config, workspace, 0);
}

} // namespace scan
} // namespace cis565
