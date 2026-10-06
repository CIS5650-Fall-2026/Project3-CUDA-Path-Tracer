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

        __global__ void kernUpsweep(int n, int offset, int stride, int* data) {
          unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

          if (idx >= n / stride) {
            return;
          }

          int right_idx = (idx + 1) * stride - 1;
          int left_idx = right_idx - offset;
          data[right_idx] += data[left_idx];
        }

        __global__ void kernDownsweep(int n, int offset, int stride, int* data) {
          unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

          if (idx >= n / stride) {
            return;
          }

          int right_idx = (idx + 1) * stride - 1;
          int left_idx = right_idx - offset;
          
          int temp = data[left_idx];
          data[left_idx] = data[right_idx];
          data[right_idx] += temp;
        }

        void deviceScan(int n, int* dev_data) {
          int upRoundedN = 1 << ilog2ceil(n);

          int blockSize = 128;

          // upsweep
          for (int d = 0; d < ilog2ceil(n); d++) {
            int offset = 1 << d;     
            int stride = 1 << (d + 1);

            int blocksPerGrid = (upRoundedN / stride + blockSize - 1) / blockSize;

            kernUpsweep<<<blocksPerGrid, blockSize>>>(upRoundedN, offset, stride, dev_data);
          }

          // downsweep
          cudaMemset(&dev_data[upRoundedN - 1], 0, sizeof(int));
          for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
            int offset = 1 << d;
            int stride = 1 << (d + 1);

            int blocksPerGrid = (upRoundedN / stride + blockSize - 1) / blockSize;

            kernDownsweep<<<blocksPerGrid, blockSize>>>(upRoundedN, offset, stride, dev_data);
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          // TODO
          // device array
          int *dev_data;
          int upRoundedN = 1 << ilog2ceil(n);

          cudaMalloc((void**)&dev_data, upRoundedN * sizeof(int));
          cudaMemset(dev_data, 0, upRoundedN * sizeof(int));
          cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          timer().startGpuTimer();
          deviceScan(n, dev_data);
          timer().endGpuTimer();

          cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

          // free device array
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
          // device arrays
          int *dev_idata, *dev_odata, *dev_indices, *dev_bools;
          int upRoundedN = 1 << ilog2ceil(n);

          cudaMalloc((void**)&dev_idata, n * sizeof(int));
          cudaMalloc((void**)&dev_odata, n * sizeof(int));

          cudaMalloc((void**)&dev_indices, upRoundedN * sizeof(int));
          cudaMalloc((void**)&dev_bools, upRoundedN * sizeof(int));

          cudaMemset(dev_bools, 0, upRoundedN * sizeof(int));
          cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          int blockSize = 128;
          int blocksPerGrid = (n + blockSize - 1) / blockSize;

          timer().startGpuTimer();

          StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);

          cudaMemcpy(dev_indices, dev_bools, upRoundedN * sizeof(int), cudaMemcpyDeviceToDevice);
          deviceScan(upRoundedN, dev_indices);

          StreamCompaction::Common::kernScatter<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
          
          int scanElems, lastElem;
          cudaMemcpy(&scanElems, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
          
          cudaMemcpy(&lastElem, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

          // free device arrays
          cudaFree(dev_idata);
          cudaFree(dev_odata);
          cudaFree(dev_bools);
          cudaFree(dev_indices);

          return scanElems + lastElem;
        }
    }
}
