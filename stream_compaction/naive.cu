#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "common.h"
#include "naive.h"

#define blockSize 16
dim3 threadsPerBlock(blockSize);

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelScan(int n, int d, int *dev_odata, int *dev_idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }
            int skip = 1 << (d - 1);
            if (k >= skip) {
                dev_odata[k] = dev_idata[k - skip] + dev_idata[k];
            } else {
                dev_odata[k] = dev_idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //allocate memory
            int *dev_idata, *dev_odata;
            cudaMalloc((void**) &dev_idata, n * sizeof(int));
            cudaMalloc((void**) &dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            //initialize parameters
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int numIter = ilog2ceil(n);

            //run the kernel scan d times
            timer().startGpuTimer();
            for (int d = 1; d <= numIter; d++) {
                kernelScan<<<fullBlocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            //shift all values to make exclusive
            for (int i = n - 1; i > 0; i--) {
              odata[i] = odata[i - 1];
            }
            odata[0] = 0;
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
