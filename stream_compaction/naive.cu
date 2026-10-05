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
        // TODO: __global__

        __global__ void kernNaiveScan(int n, int offset, int* odata, const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n)
            {
                if (index >= offset)
                {
                    odata[index] = idata[index] + idata[index - offset];
                }
                else
                {
                    odata[index] = idata[index];
                }
            }
        }

        __global__ void kernShift(int n, int* odata, const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n)
            {
                if (index == 0)
                {
                    odata[index] = 0;
                }
                else
                {
                    odata[index] = idata[index - 1];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

            if (n == 0)
            {
                return;
            }

            int* dev_a;
            int* dev_b;

            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));

            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;

            int* src = dev_a;
            int* dst = dev_b;

            timer().startGpuTimer();

            for (int offset = 1; offset < n; offset *= 2)
            {
                kernNaiveScan << <blocks, blockSize >> > (n, offset, dst, src);

                int* temp = src;
                src = dst;
                dst = temp;
            }

            kernShift << <blocks, blockSize >> > (n, dst, src);

            timer().endGpuTimer();

            cudaMemcpy(odata, dst, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}
