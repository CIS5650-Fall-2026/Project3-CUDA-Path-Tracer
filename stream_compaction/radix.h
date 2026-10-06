/**
 * @file      radix.h
 * @brief     GPU radix sort built on the work-efficient scan
 * @authors   Xuan Zhu
 * @date      2026
 * @copyright University of Pennsylvania
 */

#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int n, int* odata, const int* idata);
    }
}