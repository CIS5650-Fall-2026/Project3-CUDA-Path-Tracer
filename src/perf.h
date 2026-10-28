#pragma once

// Console timing of the path tracer: paths alive per bounce (first iteration only),
// average ms/iteration every 100 iterations, and how those ms split across the
// pipeline stages. 0 compiles every function below down to nothing
#define PERF_LOG 1

// The stages of one iteration, in the order they are issued
enum PerfStage
{
    PERF_GENERATE,
    PERF_INTERSECT,
    PERF_SORT,
    PERF_SHADE,
    PERF_COMPACT,
    PERF_GATHER,
    PERF_STAGES     // count, and the mark that closes the last stage
};

// Call once at the start of an iteration
void perfBegin();

// Call right before a stage is issued, and with PERF_STAGES after the last one
// A mark goes right after the previous launch, not after any sync; the event waits
// its turn on the GPU. A stage that is compiled out gets no mark
void perfMark(PerfStage stage);

// Call after the first iteration's compaction, so the survivors can be printed
void perfPathsAlive(int iter, int depth, int numPaths);

// Call once the iteration's GPU work has completed; prints every 100 iterations
void perfEnd(int iter);
