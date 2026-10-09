#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        // this is changeable externally for testing purposes
        int BLOCK_SIZE = 128;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // upsweep and downseep (on 2^(d + 1) strides)
        // thread i -> i * stride part
        __global__ void kernUpSweep(int numThreads, int stride, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > numThreads) {
                return;
            }

            int seg = index * stride;
            data[seg + stride - 1] += data[seg + (stride >> 1) - 1];
        }

        __global__ void kernDownSweep(int numThreads, int stride, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > numThreads) {
                return;
            }

            int seg = index * stride;
            int l = seg + (stride >> 1) - 1;
            int r = seg + stride - 1;

            int temp = data[l];
            data[l] = data[r];
            data[r] += temp;
        }

        // for both scan and compact, this is an exclusive in place scan
        // for sake of optimization, attempt to reduce number of launches
        // since top levels/layers have few active threads, they can be put
        // in one block using syncthreads between layers.
        __global__ void kernScanReduce(int size, int firstLayer, int layers, int* data) {
            int index = threadIdx.x;

            // upsweep
            for (int i = firstLayer; i < layers; ++i) {
                int stride = 1 << (i + 1);
                if (index < (size >> (i + 1))) {
                    int seg = index * stride;
                    data[seg + stride - 1] += data[seg + (stride >> 1) - 1];
                }

                __syncthreads();
            }

            if (index == 0) {
                data[size - 1] = 0;
            }
            __syncthreads();

            // downsweep
            for (int i = layers - 1; i >= firstLayer; --i) {
                int stride = 1 << (i + 1);
                if (index < (size >> (i + 1))) {
                    int seg = index * stride;
                    int l = seg + (stride >> 1) - 1;
                    int r = seg + stride - 1;
                    
                    int temp = data[l];
                    data[l] = data[r];
                    data[r] += temp;
                }
                __syncthreads();
            }

        }

        // the idea is that every layer that has less than block size active threads will happen in one block,
        // then do the remaining layers
        void scanDev(int size, int* dev_data) {
            int layers = ilog2ceil(size);

            int firstLayer = 0;
            for (; firstLayer < layers && (size >> (firstLayer + 1)) > BLOCK_SIZE; ++firstLayer) {
                int stride = 1 << (firstLayer + 1);
                int numThreads = size / stride;
                dim3 blocks((numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);
                
                kernUpSweep << <blocks, BLOCK_SIZE >> > (numThreads, stride, dev_data);
            }

            kernScanReduce << <1, BLOCK_SIZE >> > (size, firstLayer, layers, dev_data);

            for (int i = firstLayer - 1; i >= 0; --i) {
                int stride = 1 << (i + 1);
                int numThreads = size / stride;
                dim3 blocks((numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);

                kernDownSweep << <blocks, BLOCK_SIZE >> > (numThreads, stride, dev_data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int size = 1 << ilog2ceil(n);
            int* dev_data = nullptr;
            cudaMalloc((void**)&dev_data, size * sizeof(int));
            cudaMemset(dev_data, 0, size * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            
            scanDev(size, dev_data);

            timer().endGpuTimer();

            // free memory
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
            /*
                1. After allocating, map to booleans w/ kernMapToBoolean
                (this is one thread per element, recall 0s or 1s)
                2. Copy bools into indices (device -> device) in place scanning
                Avoid modifying original bools (for kernScatter later)
                3. Exclusive in place scan on indices
                4. kernScatter (also one thread per element)
                5. add indices[n - 1] and bools[n - 1] (because if last element is kept)
            */

            if (n <= 0) {
                return 0;
            }

            int size = 1 << ilog2ceil(n);
            int* dev_idata = nullptr;
            int* dev_odata = nullptr;
            int* dev_bools = nullptr;
            int* dev_indices = nullptr;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, size * sizeof(int));
            cudaMalloc((void**)&dev_indices, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_bools, 0, size * sizeof(int));
            dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();
            
            Common::kernMapToBoolean << <blocks, BLOCK_SIZE >> > (n, dev_bools, dev_idata);
            
            cudaMemcpy(dev_indices, dev_bools, size * sizeof(int), cudaMemcpyDeviceToDevice);
            scanDev(size, dev_indices);
            
            Common::kernScatter << <blocks, BLOCK_SIZE >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            timer().endGpuTimer();

            int lastIndex = 0;
            int lastBool = 0;
            
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int num = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, num * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return num;
        }
    }
}
