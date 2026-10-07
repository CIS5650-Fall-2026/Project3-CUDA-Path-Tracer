#pragma once

#include "scene.h"
#include "utilities.h"

void setMaterialSort(bool enabled);
void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void setRussianRoulettePathTerm(bool enabled);
void setMeshBoundsCulling(bool enabled);
bool getBoundsCullingEnabled();
void setMeshBoundingVolHier(bool enabled);
bool getMeshBoundingVolHier();
