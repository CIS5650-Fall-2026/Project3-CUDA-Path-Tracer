#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"

#define blockSize 128
#define ELEMENTS_PER_BLOCK (blockSize * 2)

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)  //#define CONFLICT_FREE_OFFSET(n) (0) 

namespace StreamCompaction {
    namespace Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // scan one block worth of data entirely inside shared memory
        __global__ void kernScanBlock(int n, int* odata, const int* idata, int* blockSums) {
            extern __shared__ int temp[];

            int tid = threadIdx.x;
            int base = blockIdx.x * ELEMENTS_PER_BLOCK;

            int ai = tid;
            int bi = tid + blockSize;
            int offsetA = CONFLICT_FREE_OFFSET(ai);
            int offsetB = CONFLICT_FREE_OFFSET(bi);

            temp[ai + offsetA] = (base + ai < n) ? idata[base + ai] : 0;
            temp[bi + offsetB] = (base + bi < n) ? idata[base + bi] : 0;

            int offset = 1;

            for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (tid < d) {
                    int a = offset * (2 * tid + 1) - 1;
                    int b = offset * (2 * tid + 2) - 1;
                    a += CONFLICT_FREE_OFFSET(a);
                    b += CONFLICT_FREE_OFFSET(b);
                    temp[b] += temp[a];
                }
                offset *= 2;
            }

            if (tid == 0) {
                int last = ELEMENTS_PER_BLOCK - 1;
                last += CONFLICT_FREE_OFFSET(last);
                if (blockSums != nullptr) {
                    blockSums[blockIdx.x] = temp[last];
                }
                temp[last] = 0;
            }

            for (int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (tid < d) {
                    int a = offset * (2 * tid + 1) - 1;
                    int b = offset * (2 * tid + 2) - 1;
                    a += CONFLICT_FREE_OFFSET(a);
                    b += CONFLICT_FREE_OFFSET(b);
                    int t = temp[a];
                    temp[a] = temp[b];
                    temp[b] += t;
                }
            }
            __syncthreads();

            if (base + ai < n) {
                odata[base + ai] = temp[ai + offsetA];
            }
            if (base + bi < n) {
                odata[base + bi] = temp[bi + offsetB];
            }
        }

        // add each block's scanned sum back into every element of that block
        __global__ void kernAddBlockSums(int n, int* data, const int* blockSums) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            data[index] += blockSums[index / ELEMENTS_PER_BLOCK];
        }

        // recursive device-side scan, no timer
        void scanDeviceShared(int n, int* dev_data) {
            int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
            int sharedBytes = (ELEMENTS_PER_BLOCK + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK)) * sizeof(int);

            if (numBlocks == 1) {
                kernScanBlock << <1, blockSize, sharedBytes >> > (n, dev_data, dev_data, nullptr);
                return;
            }

            int* dev_blockSums;
            cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed");

            kernScanBlock << <numBlocks, blockSize, sharedBytes >> > (n, dev_data, dev_data, dev_blockSums);

            scanDeviceShared(numBlocks, dev_blockSums);

            dim3 addBlocks((n + blockSize - 1) / blockSize);
            kernAddBlockSums << <addBlocks, blockSize >> > (n, dev_data, dev_blockSums);

            cudaFree(dev_blockSums);
        }

        void scan(int n, int* odata, const int* idata) {
            int* dev_data;
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            scanDeviceShared(n, dev_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy back failed");

            cudaFree(dev_data);
        }
    }
}