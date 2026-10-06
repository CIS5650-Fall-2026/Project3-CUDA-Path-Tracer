/**
 * @file      radix.cu
 * @brief     GPU radix sort built on the work-efficient scan
 * @authors   Xuan Zhu
 * @date      2026
 * @copyright University of Pennsylvania
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"
#include <algorithm>

#define blockSize 128

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // e[i] = 1 when the current bit of idata[i] is 0
        __global__ void kernComputeE(int n, int bit, int* e, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            e[index] = ((idata[index] >> bit) & 1) ? 0 : 1;
        }

        // scatter each element to its position for this bit
        __global__ void kernRadixScatter(int n, int totalFalses, int bit,
            int* odata, const int* idata, const int* f) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int bitIsOne = (idata[index] >> bit) & 1;
            int dest = bitIsOne ? (index - f[index] + totalFalses) : f[index];
            odata[dest] = idata[index];
        }

        /**
         * Radix sort for non-negative integers, built on the work-efficient scan.
         */
        void sort(int n, int* odata, const int* idata) {
            int paddedN = 1 << ilog2ceil(n);

            int* dev_in;
            int* dev_out;
            int* dev_e;
            int* dev_f;

            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMalloc((void**)&dev_f, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc failed");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // only sort up to the highest bit actually present in the data
            int maxVal = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] > maxVal) {
                    maxVal = idata[i];
                }
            }
            int numBits = ilog2ceil(maxVal + 1);

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int bit = 0; bit < numBits; bit++) {
                kernComputeE << <fullBlocks, blockSize >> > (n, bit, dev_e, dev_in);

                // scan e into f, the scan buffer is padded so zero it first
                cudaMemset(dev_f, 0, paddedN * sizeof(int));
                cudaMemcpy(dev_f, dev_e, n * sizeof(int), cudaMemcpyDeviceToDevice);
                StreamCompaction::Efficient::scanDevice(paddedN, dev_f);

                // totalFalses = f[n-1] + e[n-1], same trick as in compact
                int lastF = 0;
                int lastE = 0;
                cudaMemcpy(&lastF, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastE, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalses = lastF + lastE;

                kernRadixScatter << <fullBlocks, blockSize >> > (n, totalFalses, bit,
                    dev_out, dev_in, dev_f);

                std::swap(dev_in, dev_out);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy back failed");

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_e);
            cudaFree(dev_f);
        }
    }
}