#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>
#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        // one level of the naive scan: add the element offset positions to the left
        __global__ void kernNaiveScanStep(int n, int offset, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                // still need to carry the value over, we are ping-ponging buffers
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* dev_A;
            int* dev_B;
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_A failed");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_B failed");

            cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("memcpy to dev_A failed");

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int offset = 1; offset < n; offset *= 2) {
                kernNaiveScanStep << <fullBlocks, blockSize >> > (n, offset, dev_B, dev_A);
                std::swap(dev_A, dev_B);
            }

            timer().endGpuTimer();

            // the naive algorithm gives an inclusive scan, shift right by one to make it exclusive
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_A, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy back to odata failed");

            cudaFree(dev_A);
            cudaFree(dev_B);
        }
    }
}
