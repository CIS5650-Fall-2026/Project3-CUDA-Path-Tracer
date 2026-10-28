#include "perf.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

#if PERF_LOG

// Per-stage timing. An event is recorded on the GPU right before every stage is issued,
// plus one after the last stage; once the iteration is done, the gap between each pair
// of consecutive events is added to the stage that ran between them. Events cost
// nothing on the host and need no extra sync
static const char* stageName[PERF_STAGES] = { "generate", "intersect", "sort", "shade", "compact", "gather" };
static double stageMs[PERF_STAGES];
static std::vector<cudaEvent_t> events;   // events[0] marks the start of the iteration
static std::vector<int> markStage;        // markStage[k] is the stage that starts at events[k + 1]

// whole-iteration time on the host, from perfBegin to perfEnd
static double accumMs = 0.0;
static int frameCount = 0;
static std::chrono::high_resolution_clock::time_point start;

// events are created on first use and kept
static cudaEvent_t eventAt(size_t index)
{
    while (events.size() <= index)
    {
        cudaEvent_t e;
        cudaEventCreate(&e);
        events.push_back(e);
    }
    return events[index];
}

void perfBegin()
{
    start = std::chrono::high_resolution_clock::now();
    markStage.clear();
    cudaEventRecord(eventAt(0));
}

void perfMark(PerfStage stage)
{
    cudaEventRecord(eventAt(markStage.size() + 1));
    markStage.push_back(stage);
}

void perfPathsAlive(int iter, int depth, int numPaths)
{
    if (iter == 1)
    {
        printf("[perf] bounce %d: %d paths alive\n", depth, numPaths);
    }
}

void perfEnd(int iter)
{
    const auto end = std::chrono::high_resolution_clock::now();
    accumMs += std::chrono::duration<double, std::milli>(end - start).count();
    frameCount++;

    // mark k opens its stage at events[k + 1] and the next mark closes it, so the last
    // mark, the PERF_STAGES sentinel, has no interval of its own. The memset before each
    // intersection lands in the stage that precedes it, which is well under 0.1 ms
    for (size_t k = 0; k + 1 < markStage.size(); k++)
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, events[k + 1], events[k + 2]);
        stageMs[markStage[k]] += ms;
    }

    if (frameCount == 100)
    {
        printf("[perf] iter %d: avg %.2f ms/iteration (%.1f FPS) over last 100 iterations\n",
            iter, accumMs / frameCount, 1000.0 * frameCount / accumMs);

        // one line the timing scripts parse: name and average ms, alternating
        printf("[perf] stages");
        for (int st = 0; st < PERF_STAGES; st++)
        {
            printf(" %s %.2f", stageName[st], stageMs[st] / frameCount);
            stageMs[st] = 0.0;
        }
        printf("\n");

        accumMs = 0.0;
        frameCount = 0;
    }
}

#else

void perfBegin() {}
void perfMark(PerfStage) {}
void perfPathsAlive(int, int, int) {}
void perfEnd(int) {}

#endif
