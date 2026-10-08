#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "pathTraceCommon.h"
#include "camera.h"
#include "scene.h"
#include "environmentMap.h"
#include "renderSettings.h"

#include <memory>

void launchCameraRayGenKernel(PathState* dev_pathStates,
	int width, int height,
	const Camera& camera,
	int frameIndex,
	bool bMSAAEnabled);

void launchIntersectKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData, 
	const std::unique_ptr<Scene>& scene,
	int activePathCount);

void launchShadeKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	const std::unique_ptr<Scene>& scene,
	const std::unique_ptr<EnvironmentMap>& environmentMap,
	int activePathCount,
	glm::vec3* dev_accumulatedColor,
	glm::vec4* dev_currentDirectColor,
	glm::vec4* dev_currentIndirectColor,
	unsigned int* dev_sampleCounts,
	LightSamplingMode lightSamplingMode,
	float minRussianRouletteSurvival,
	int iteration,
	int frameIndex);

void launchRecordActivePathsKernel(PathState* dev_pathStates,
	int activePathCount,
	glm::vec3* dev_accumulatedColor,
	glm::vec4* dev_currentDirectColor,
	glm::vec4* dev_currentIndirectColor,
	unsigned int* dev_sampleCounts);

void launchDisplayAccumulatedSamplesKernel(cudaSurfaceObject_t surface,
	glm::vec3* dev_accumulatedColor,
	unsigned int* dev_sampleCounts,
	int width, int height);

void launchDebugUVKernel(IntersectionData* dev_intersectionData, cudaSurfaceObject_t surface, int width, int height);

// Draw bvh heatmap to the surface. Should be called after the first intersect kernel.
void launchBvhHeatmapKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	int activePathCount,
	cudaSurfaceObject_t surface,
	int width,
	BvhHeatmapMode mode,
	int maxSteps);

int runStreamCompaction(PathState* dev_pathStates, int numActivePaths);

void sortPathsByMaterial(PathState* dev_pathStates, IntersectionData* dev_intersectionData, int activePathCount, const Material* dev_materials);