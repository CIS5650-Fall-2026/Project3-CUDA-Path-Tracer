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
        
        __global__ void kernNaiveScan(int n, int offset, int* odata, const int* idata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= n) {
              return;
            }

            if (idx >= offset) {
              odata[idx] = idata[idx - offset] + idata[idx];
            }
            else {
              odata[idx] = idata[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {  
            // TODO
            // device arrays
            int *dev_idata, *dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocksPerGrid = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); d++) {
              int offset = 1 << (d - 1);
              kernNaiveScan<<<blocksPerGrid, blockSize>>>(n, offset, dev_odata, dev_idata);
              std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            
            // data is always in idata
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // convert from inclusive to exclusive scan
            for (int i = n - 1; i > 0; i--) {
              odata[i] = odata[i - 1];
            }
            odata[0] = 0;

            // free device arrays
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
