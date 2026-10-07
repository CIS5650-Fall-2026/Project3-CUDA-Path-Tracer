#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 16

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void reduce(int threads, int n, int d, int *dev_idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= threads) {
                return;
            }
            int skip = 1 << (d + 1);
            int halfSkip = 1 << d;
            int k = index * skip;
            if ((k + skip - 1) == n - 1) {
                //set last element to 0
                dev_idata[k + skip - 1] = 0;
            } else {
                dev_idata[k + skip - 1] = dev_idata[k + halfSkip - 1] + dev_idata[k + skip - 1];
            }
        }

        __global__ void expand(int threads, int d, int* dev_idata) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= threads) {
                return;
            }

            int skip = 1 << (d + 1);
            int halfSkip = 1 << d;

            int k = index * skip;

            int temp = dev_idata[k + halfSkip - 1];
            dev_idata[k + halfSkip - 1] = dev_idata[k + skip - 1];
            dev_idata[k + skip - 1] = temp + dev_idata[k + skip - 1];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool timerRunning) {
            int numIter = ilog2ceil(n);
            int paddedN = 1 << numIter;
            int *dev_idata;
            cudaMalloc((void**) &dev_idata, paddedN * sizeof(int));
            //init dev_idata with paddedN elements
            cudaMemset(dev_idata, 0, paddedN * sizeof(int));
            //copy n elements from idata (remaining are 0 in dev_idata)
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            if (!timerRunning) {
                timer().startGpuTimer();
            }

            //upsweep
            for (int d = 0; d < numIter; d++) {
                int threads = paddedN >> (d + 1);
                int blocks = (threads + blockSize - 1) / blockSize;
                reduce<<<blocks, blockSize>>>(threads, paddedN, d, dev_idata);
            }

            //downsweep
            for (int d = numIter - 1; d >= 0; d--) {
                int threads = paddedN >> (d + 1);
                int blocks = (threads + blockSize - 1) / blockSize;
                expand<<<blocks, blockSize >>> (threads, d, dev_idata);
            }

            if (!timerRunning) {
                timer().endGpuTimer();
            }
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
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
            int blocks = (n + blockSize - 1) / blockSize;
            int *bools;
            int *dev_idata;
            int *dev_odata;
            int *indices;
            cudaMalloc((void**) &bools, n * sizeof(int));
            cudaMalloc((void**) &dev_idata, n * sizeof(int));
            cudaMalloc((void**) &indices, n * sizeof(int));
            cudaMalloc((void**) &dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, bools, dev_idata);
            int *host_bools = new int[n];
            int *host_indices = new int[n];
            cudaMemcpy(host_bools, bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            scan(n, host_indices, host_bools, true);
            cudaMemcpy(indices, host_indices, n * sizeof(int), cudaMemcpyHostToDevice);
            StreamCompaction::Common::kernScatter<<<blocks,blockSize>>>(n, dev_odata, dev_idata, bools, indices);
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int last_bool = 0;
            int last_index = 0;
            cudaMemcpy(&last_bool, &bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_index, &indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            int numElements = last_index + last_bool;
            delete[] host_bools;
            delete[] host_indices;
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(bools);
            cudaFree(indices);
            timer().endGpuTimer();
            return numElements;
        }
    }
}
