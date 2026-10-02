#pragma once

#include "common.h"

struct PathSegment;

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scanDevice(int n, int* data);

        void scan(int n, int* odata, const int* idata);

        int compact(int n, int* odata, const int* idata);

        // Device pointers; scratch holds at least 3*n+1024 ints. Input/output must not alias.
        int compact(int n, PathSegment* odata, const PathSegment* idata, int* scratch);
    }
}