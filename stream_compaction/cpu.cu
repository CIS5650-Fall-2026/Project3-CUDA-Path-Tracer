#include <cstdio>
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

        // timer-free version so compactWithScan can reuse it
        // (calling scan() directly would start the timer twice)
        void scanImpl(int n, int* odata, const int* idata) {
            if (n <= 0) return;
            odata[0] = 0;                                // exclusive scan starts at 0
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];  // sum of everything before i
            }
        }




        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanImpl(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int count = 0;
            timer().startCpuTimer();
          
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n <= 0) return 0;

            // allocate outside the timer, we only want to measure the algorithm
            int* bools = new int[n];
            int* indices = new int[n];
            

            timer().startCpuTimer();

            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            scanImpl(n, indices, bools);

            for (int i = 0; i < n; i++) {
                if (bools[i]) {
                    odata[indices[i]] = idata[i];
                }
            }

            // exclusive scan drops the last element, so add it back
            int count = indices[n - 1] + bools[n - 1];

            timer().endCpuTimer();
            delete[] bools;
            delete[] indices;
            return count;
        }
    }
}
