#pragma once

#include "../sceneStructs.h"

#include <glm/glm.hpp>

void CreateTestNVDB(glm::mat4* invTransform, void** d_grid);

bool LoadNVDB(const std::string& fileName, glm::mat4* invTransform, void** d_grid, std::string gridName);