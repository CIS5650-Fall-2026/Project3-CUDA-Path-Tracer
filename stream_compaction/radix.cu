#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"
#include "radix.h"

namespace StreamCompaction {

    namespace Radix {

        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMapBit(
            int n,
            int bit,
            int* bools,
            const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {
                unsigned int key =
                    static_cast<unsigned int>(idata[index]) ^ 0x80000000u;

                bools[index] = ((key >> bit) & 1u) == 0u ? 1 : 0;
            }
        }

        __global__ void kernRadixScatter(
            int n,
            int* odata,
            const int* idata,
            const int* bools,
            const int* indices)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {
                int totalZeros =
                    indices[n - 1] + bools[n - 1];

                int outputIndex;

                if (bools[index] == 1) {
                    outputIndex = indices[index];
                }
                else {
                    outputIndex =
                        totalZeros + index - indices[index];
                }

                odata[outputIndex] = idata[index];
            }
        }

        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }

            int paddedN = 1 << ilog2ceil(n);

            int* dev_input;
            int* dev_output;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc(
                (void**)&dev_input,
                n * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_output,
                n * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_bools,
                paddedN * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_indices,
                paddedN * sizeof(int)
            );

            cudaMemcpy(
                dev_input,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            cudaMemset(
                dev_bools,
                0,
                paddedN * sizeof(int)
            );

            const int blockSize = 256;
            const int numBlocks =
                (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; ++bit) {
                kernMapBit << <numBlocks, blockSize >> > (
                    n,
                    bit,
                    dev_bools,
                    dev_input
                    );

                cudaMemcpy(
                    dev_indices,
                    dev_bools,
                    paddedN * sizeof(int),
                    cudaMemcpyDeviceToDevice
                );

                StreamCompaction::Efficient::scanDevice(
                    paddedN,
                    dev_indices
                );

                kernRadixScatter << <numBlocks, blockSize >> > (
                    n,
                    dev_output,
                    dev_input,
                    dev_bools,
                    dev_indices
                    );

                int* temp = dev_input;
                dev_input = dev_output;
                dev_output = temp;
            }

            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                dev_input,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(dev_input);
            cudaFree(dev_output);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            checkCUDAError("Radix sort failed");
        }

    }

}