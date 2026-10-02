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

        __global__ void kernNaiveScan(
            int n,
            int offset,
            int* odata,
            const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            if (index >= offset) {
                odata[index] =
                    idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata,
         * storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            if (n <= 0) {
                return;
            }

            int* dev_input;
            int* dev_output;

            cudaMalloc((void**)&dev_input, n * sizeof(int));
            cudaMalloc((void**)&dev_output, n * sizeof(int));

            cudaMemset(dev_input, 0, sizeof(int));

            if (n > 1) {
                cudaMemcpy(
                    dev_input + 1,
                    idata,
                    (n - 1) * sizeof(int),
                    cudaMemcpyHostToDevice
                );
            }

            const int blockSize = 128;
            const int numBlocks =
                (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            for (int offset = 1; offset < n; offset *= 2) {

                kernNaiveScan << <numBlocks, blockSize >> > (
                    n,
                    offset,
                    dev_output,
                    dev_input
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

            checkCUDAError("Naive scan failed");
        }

    }

}