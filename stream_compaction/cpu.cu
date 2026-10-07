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

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata, bool timerRunning) {
            if (!timerRunning) {
                timer().startCpuTimer();
            }
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i-1] + odata[i-1];
            }
            if (!timerRunning) {
                timer().endCpuTimer();
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int odataCount = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[odataCount] = idata[i];
                    odataCount++;
                }
            }
            timer().endCpuTimer();
            return odataCount;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *temp1 = new int[n];
            int *temp2 = new int[n];
            int odataIndex = 0;

            timer().startCpuTimer();
            //prepare temp1 array
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    temp1[i] = 1;
                } else {
                    temp1[i] = 0;
                }
            }

            //exclusive scan temp1 array to get temp2 (indices for output array)
            scan(n, temp2, temp1, true);

            //if temp1 was 1, then put the input data in the temp2[i]'th element of odata
            for (int i = 0; i < n; i++) {
                if (temp1[i] == 1) {
                    odataIndex = temp2[i];
                    odata[odataIndex] = idata[i];
                }
            }
            timer().endCpuTimer();

            //add 1 to odataIndex to get number of elements in compaction
            int numElements = temp2[n - 1] + temp1[n - 1];

            //cleanup
            delete[] temp1;
            delete[] temp2;

            return numElements;
        }
    }
}
