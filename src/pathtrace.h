#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

struct BounceStatistics
{
    int depth = 0, raysIn = 0, raysAlive = 0, raysScheduled = 0;
    float cameraMs = 0, intersectionMs = 0, sortingMs = 0;
    float shadingMs = 0, gatherMs = 0, compactionMs = 0;
};

void enableTraceStatistics(bool enabled);
const std::vector<BounceStatistics>& lastBounceStatistics();
