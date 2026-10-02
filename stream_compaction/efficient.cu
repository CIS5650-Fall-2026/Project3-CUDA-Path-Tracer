#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"
#include "../src/sceneStructs.h"

namespace StreamCompaction {

    namespace Efficient {

        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int offset, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int dataIndex = (index + 1) * offset * 2 - 1;

            if (dataIndex < n) {
                data[dataIndex] += data[dataIndex - offset];
            }
        }

        __global__ void kernDownSweep(int n, int offset, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int dataIndex = (index + 1) * offset * 2 - 1;

            if (dataIndex < n) {
                int temp = data[dataIndex - offset];
                data[dataIndex - offset] = data[dataIndex];
                data[dataIndex] += temp;
            }
        }

        __global__ void kernSetZero(int n, int* data) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                data[n - 1] = 0;
            }
        }

        void scanDevice(int n, int* data) {
            const int blockSize = 512;

            // Up-sweep
            for (int offset = 1; offset < n; offset *= 2) {
                int numThreads = n / (offset * 2);
                int numBlocks = (numThreads + blockSize - 1) / blockSize;

                kernUpSweep << <numBlocks, blockSize >> > (
                    n,
                    offset,
                    data
                    );
            }

            // Clear root for exclusive scan
            kernSetZero << <1, 1 >> > (
                n,
                data
                );

            // Down-sweep
            for (int offset = n / 2; offset >= 1; offset /= 2) {
                int numThreads = n / (offset * 2);
                int numBlocks = (numThreads + blockSize - 1) / blockSize;

                kernDownSweep << <numBlocks, blockSize >> > (
                    n,
                    offset,
                    data
                    );
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }

            int paddedN = 1 << ilog2ceil(n);
            int* dev_data;

            cudaMalloc(
                (void**)&dev_data,
                paddedN * sizeof(int)
            );

            cudaMemset(
                dev_data,
                0,
                paddedN * sizeof(int)
            );

            cudaMemcpy(
                dev_data,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            timer().startGpuTimer();

            scanDevice(
                paddedN,
                dev_data
            );

            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                dev_data,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(dev_data);

            checkCUDAError("Work-efficient scan failed");
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
        int compact(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc(
                (void**)&dev_idata,
                n * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_odata,
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
                dev_idata,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            cudaMemset(
                dev_bools,
                0,
                paddedN * sizeof(int)
            );

            cudaMemset(
                dev_indices,
                0,
                paddedN * sizeof(int)
            );

            const int blockSize = 256;
            const int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // Map
            StreamCompaction::Common::kernMapToBoolean << <numBlocks, blockSize >> > (
                n,
                dev_bools,
                dev_idata
                );

            cudaMemcpy(
                dev_indices,
                dev_bools,
                paddedN * sizeof(int),
                cudaMemcpyDeviceToDevice
            );

            // Scan
            scanDevice(
                paddedN,
                dev_indices
            );

            // Scatter
            StreamCompaction::Common::kernScatter << <numBlocks, blockSize >> > (
                n,
                dev_odata,
                dev_idata,
                dev_bools,
                dev_indices
                );

            timer().endGpuTimer();

            int lastIndex;
            int lastBool;

            cudaMemcpy(
                &lastIndex,
                dev_indices + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                &lastBool,
                dev_bools + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            int count = lastIndex + lastBool;

            cudaMemcpy(
                odata,
                dev_odata,
                count * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            checkCUDAError("Work-efficient compaction failed");

            return count;
        }

        // Device-resident path compaction uses a recursive shared-memory scan.
        constexpr int scanSize = 512;
        __global__ void kernMapPathsToBoolean(const PathSegment *p, int *flags, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                flags[i] = p[i].remainingBounces > 0 ? 1 : 0;
        }
        // Blelloch work-efficient exclusive scan, two elements per thread in shared memory.
        __global__ void kernScanBlocks(const int *input, int *output, int *totals, int n)
        {
            __shared__ int values[scanSize];
            int tid = threadIdx.x, start = blockIdx.x * scanSize;
            values[tid] = start + tid < n ? input[start + tid] : 0;
            values[tid + scanSize / 2] = start + tid + scanSize / 2 < n ? input[start + tid + scanSize / 2] : 0;
            __syncthreads();
            for (int stride = 1; stride < scanSize; stride *= 2)
            {
                int i = (tid + 1) * stride * 2 - 1;
                if (i < scanSize)
                    values[i] += values[i - stride];
                __syncthreads();
            }
            if (tid == 0)
            {
                totals[blockIdx.x] = values[scanSize - 1];
                values[scanSize - 1] = 0;
            }
            __syncthreads();
            for (int stride = scanSize / 2; stride >= 1; stride /= 2)
            {
                int i = (tid + 1) * stride * 2 - 1;
                if (i < scanSize)
                {
                    int a = values[i - stride];
                    values[i - stride] = values[i];
                    values[i] += a;
                }
                __syncthreads();
            }
            if (start + tid < n)
                output[start + tid] = values[tid];
            if (start + tid + scanSize / 2 < n)
                output[start + tid + scanSize / 2] = values[tid + scanSize / 2];
        }
        __global__ void kernAddOffsets(int *output, const int *offsets, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                output[i] += offsets[i / scanSize];
        }
        void scanDevice(const int *input, int *output, int n, int *workspace)
        {
            int blocks = (n + scanSize - 1) / scanSize;
            int *totals = workspace;
            int *offsets = workspace + blocks;
            kernScanBlocks<<<blocks, scanSize / 2>>>(input, output, totals, n);
            if (blocks > 1)
            {
                scanDevice(totals, offsets, blocks, workspace + 2 * blocks);
                kernAddOffsets<<<(n + 255) / 256, 256>>>(output, offsets, n);
            }
        }
        __global__ void kernScatterPaths(const PathSegment *input, PathSegment *output, const int *flags, const int *offsets, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n && flags[i])
                output[offsets[i]] = input[i];
        }

        int compact(int n, PathSegment *output, const PathSegment *input, int *scratch)
        {
            if (n <= 0)
                return 0;
            int *flags = scratch;
            int *offsets = scratch + n;
            kernMapPathsToBoolean<<<(n + 255) / 256, 256>>>(input, flags, n);
            scanDevice(flags, offsets, n, scratch + 2 * n);
            kernScatterPaths<<<(n + 255) / 256, 256>>>(input, output, flags, offsets, n);
            int lastFlag = 0, lastOffset = 0;
            cudaMemcpy(&lastFlag, flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastOffset, offsets + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            return lastFlag + lastOffset;
        }

    }

}