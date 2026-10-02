#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtraceRefresh(uchar4* pbo, int iteration);
struct TraceStats {
    float generateMs=0, intersectMs=0, sortMs=0, shadeMs=0, compactMs=0, gatherMs=0, postMs=0;
    std::vector<int> activePaths;
};
void pathtraceEnableStats(bool enabled);
const TraceStats& pathtraceLastStats();
