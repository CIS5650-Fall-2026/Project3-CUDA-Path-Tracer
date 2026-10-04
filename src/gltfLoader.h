 
#pragma once
#include <string>
#include <vector>
#include "sceneStructs.h";

void inspectGLTF(const std::string& filename);

void loadGLTF(
	const std::string& filename,
	const glm::mat4& objectTransform,
	std::vector<Triangle>& triangles
);