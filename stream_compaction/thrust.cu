#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        struct filter {
            __host__ __device__ bool operator()(int x) const {
                return x == 0;
            }
        };
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::device_vector<int> dv_in(idata, idata + n);
            thrust::device_vector<int> dv_out(n);
            timer().startGpuTimer();
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();
            thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }

        /**
         * Perform stream compaction using thrust
         */
        int compact(int n, int *odata, const int *idata) {
            thrust::device_vector<int> dv_in(idata, idata + n);
            timer().startGpuTimer();
            auto end = thrust::remove_if(dv_in.begin(), dv_in.end(), filter());
            timer().endGpuTimer();
            int count = end - dv_in.begin();
            thrust::copy(dv_in.begin(), end, odata);
            return count;
        }
    }
}
