#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernUpSweep(int n, int stride, int* data)
        {
            int index = (blockIdx.x * blockDim.x + threadIdx.x + 1) * stride * 2 - 1;

            if (index < n)
            {
                data[index] += data[index - stride];
            }
        }

        __global__ void kernDownSweep(int n, int stride, int* data)
        {
            int index = (blockIdx.x * blockDim.x + threadIdx.x + 1) * stride * 2 - 1;

            if (index < n)
            {
                int temp = data[index - stride];
                data[index - stride] = data[index];
                data[index] += temp;
            }
        }

        void scan(int n, int *odata, const int *idata) {

            // TODO

            if (n == 0)
            {
                return;
            }

            int paddedN = 1 << ilog2ceil(n);

            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;

            timer().startGpuTimer();

            for (int stride = 1; stride < paddedN; stride *= 2)
            {
                int threadCount = paddedN / (stride * 2);
                int blocks = (threadCount + blockSize - 1) / blockSize;

                kernUpSweep << <blocks, blockSize >> > (paddedN, stride, dev_data);
            }

            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));

            for (int stride = paddedN / 2; stride >= 1; stride /= 2)
            {
                int threadCount = paddedN / (stride * 2);
                int blocks = (threadCount + blockSize - 1) / blockSize;

                kernDownSweep << <blocks, blockSize >> > (paddedN, stride, dev_data);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            
            // TODO

            if (n == 0)
            {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);

            int* dev_input;
            int* dev_bools;
            int* dev_indices;
            int* dev_output;

            cudaMalloc((void**)&dev_input, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            cudaMalloc((void**)&dev_output, n * sizeof(int));

            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaMemset(dev_bools, 0, paddedN * sizeof(int));
            cudaMemset(dev_indices, 0, paddedN * sizeof(int));

            int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;
            
            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <blocks, blockSize >> > (n, dev_bools, dev_input);

            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int stride = 1; stride < paddedN; stride *= 2)
            {
                int threadCount = paddedN / (stride * 2);
                int scanBlocks = (threadCount + blockSize - 1) / blockSize;

                kernUpSweep << <scanBlocks, blockSize >> > (paddedN, stride, dev_indices);
            }

            cudaMemset(dev_indices + paddedN - 1, 0, sizeof(int));

            for (int stride = paddedN / 2; stride >= 1; stride /= 2)
            {
                int threadCount = paddedN / (stride * 2);
                int scanBlocks = (threadCount + blockSize - 1) / blockSize;

                kernDownSweep << <scanBlocks, blockSize >> > (paddedN, stride, dev_indices);
            }

            StreamCompaction::Common::kernScatter << <blocks, blockSize >> > (n, dev_output, dev_input, dev_bools, dev_indices);

            timer().endGpuTimer();

            int lastIndex;
            int lastBool;

            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_output, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_input);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_output);

            return count;
        }
    }
}
