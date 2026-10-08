#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "pathTraceCommon.h"
#include "scene.h"
#include "camera.h"
#include "environmentMap.h"
#include "svgfManager.h"
#include "renderSettings.h"
#include "gpuProfiler.h"

#include <memory>

class Renderer {
public:
	Renderer();
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

	void resize(vk::raii::Device& device, HANDLE sharedMemoryHandle,
		vk::Extent2D extent, size_t allocationSize);

	void render(const std::unique_ptr<Scene>& scene, const std::unique_ptr<EnvironmentMap>& environmentMap, const Camera& camera, bool bClearAccumulatedSamples);

	void saveCurrentRenderToFile(const std::string& filepath);

	void drawRenderSettingsImGui(Camera& camera);

private:
	void cleanup();

	int getPixelCount() const { return m_extent.width * m_extent.height; }

	bool hasReachedSampleTarget() const {
		return m_renderSettings.bLimitSamples && m_frameIndex >= m_renderSettings.targetSamplesPerPixel;
	}

	RenderSettings m_renderSettings;

	cudaExternalMemory_t m_cudaExtMemory = nullptr;
	cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
	cudaArray_t m_cudaArray = nullptr;
	cudaSurfaceObject_t m_cudaSurfaceObject = 0;

	vk::Extent2D m_extent;

	PathState* dev_pathStates = nullptr;
	IntersectionData* dev_intersectionData = nullptr;
	unsigned int* dev_sampleCounts = nullptr;
	glm::vec3* dev_accumulatedColor = nullptr;

	int m_frameIndex = 0;

	SVGFManager m_svgfManager;

	GpuProfiler m_profiler;
};