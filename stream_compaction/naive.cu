#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // read from idata and write to odata
        __global__ void kernNaiveScan(int n, int offset, int* odata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        // right shift and put identity in front (0)
        __global__ void kernToExclusive(int n, int* odata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n < 0) {
                return;
            }

            // later, "swap" between two dev arrays to prevent race conditions
            int* dev_a = nullptr;
            int* dev_b = nullptr;
            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 128;
            dim3 blocks((n + blockSize - 1) / blockSize);
            int p = ilog2ceil(n);
            
            timer().startGpuTimer();
            
            for (int d = 1; d <= p; ++d) {
                // k >= 2^(d - 1)
                int offset = 1 << (d - 1);
                kernNaiveScan << <blocks, blockSize >> > (n, offset, dev_b, dev_a);
                std::swap(dev_a, dev_b);
                // array a will have the finished inclusive scan, so later can write
                // inclusive -> exclusive in array b
            }
            
            kernToExclusive << <blocks, blockSize >> > (n, dev_b, dev_a);

            timer().endGpuTimer();

            // free memory!!!!
            cudaMemcpy(odata, dev_b, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}
