#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
// Always allocate. Restore uploads the already-restored host accumulation.
void pathtraceInit(Scene *scene, bool restoreAccumulation = false);
void pathtraceFree();
void pathtraceDisplay(uchar4 *pbo, int completedSamples);
struct PathTraceBounce {
    int bounce;
    int activeAfter; // -1: not counted in uncompacted mode (no timed reduction).
    int launchedPaths;
};
void pathtrace(uchar4 *pbo, int frame, int iteration,
    std::vector<PathTraceBounce>* bounceCounts = nullptr);
