#pragma once
#include "scene.h"
void saveCheckpoint(const Scene &scene, int samples, const std::string &path);
int loadCheckpoint(Scene &scene, const std::string &path);
