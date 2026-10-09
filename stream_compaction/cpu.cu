#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // CPU versions of kernMapToBoolean and kernScatter, and the scan method
        namespace {
            void scanner(int n, int* odata, const int* idata) {
                int sum = 0;
                for (int i = 0; i < n; ++i) {
                    // avoid accidentally overwriting idata
                    int temp = idata[i];
                    odata[i] = sum;
                    sum += temp;
                }
            }

            void mapToBoolean(int n, int* bools, const int* idata) {
                for (int i = 0; i < n; ++i) {
                    if (idata[i] != 0) {
                        bools[i] = 1;
                    }
                    else {
                        bools[i] = 0;
                    }
                }
            }

            void scatter(int n, int* odata, const int* idata,
                const int* bools, const int* indices) {
                for (int i = 0; i < n; ++i) {
                    if (bools[i] == 1) {
                        odata[indices[i]] = idata[i];
                    }
                }
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            scanner(n, odata, idata);

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int num = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[num++] = idata[i];
                }
            }

            timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }
            std::vector<int> bools(n);
            std::vector<int> indices(n);

            timer().startCpuTimer();

            mapToBoolean(n, bools.data(), idata);

            scanner(n, indices.data(), bools.data());

            scatter(n, odata, idata, bools.data(), indices.data());

            int num = indices[n - 1] + bools[n - 1];

            timer().endCpuTimer();

            return num;
        }
    }
}
