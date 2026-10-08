#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

__host__ __device__ glm::ivec3 finalizeColor(glm::vec3 rgb, int iter, float exposure, bool gammaCorrect);
