#include "renderer.h"

#include "intersection.h"
#include "cudaHelpers.h"
#include "tonemapping.h"

#include "stb_image_write.h"
#include "imgui.h"

#include <vector>
#include <algorithm>
#include <cmath>

#include "kernel.h"

Renderer::Renderer() :
    m_renderSettings(RenderSettings{}),
    m_svgfManager(SVGFManager())
{

}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::resize(vk::raii::Device& device, HANDLE sharedMemoryHandle,
    vk::Extent2D extent, size_t allocationSize) {
    cleanup();
    m_extent = extent;

    cudaExternalMemoryHandleDesc extMemDesc{};
    extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    extMemDesc.handle.win32.handle = sharedMemoryHandle;
    extMemDesc.size = allocationSize;

    CUDA_CHECK(cudaImportExternalMemory(&m_cudaExtMemory, &extMemDesc));

    cudaExternalMemoryMipmappedArrayDesc mipDesc{};
    mipDesc.offset = 0;
    mipDesc.formatDesc.x = 8; 
    mipDesc.formatDesc.y = 8;
    mipDesc.formatDesc.z = 8; 
    mipDesc.formatDesc.w = 8;
    mipDesc.formatDesc.f = cudaChannelFormatKindUnsigned;
    mipDesc.extent.width = m_extent.width;
    mipDesc.extent.height = m_extent.height;
    mipDesc.extent.depth = 0;
    mipDesc.numLevels = 1;

    CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&m_cudaMipmappedArray, m_cudaExtMemory, &mipDesc));

    CUDA_CHECK(cudaGetMipmappedArrayLevel(&m_cudaArray, m_cudaMipmappedArray, 0));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_cudaArray;

    CUDA_CHECK(cudaCreateSurfaceObject(&m_cudaSurfaceObject, &resDesc));

    CUDA_CHECK(cudaMalloc((void**)&dev_pathStates, getPixelCount() * sizeof(PathState)));

    CUDA_CHECK(cudaMalloc((void**)&dev_intersectionData, getPixelCount() * sizeof(IntersectionData)));

    CUDA_CHECK(cudaMalloc((void**)&dev_sampleCounts, getPixelCount() * sizeof(unsigned int)));

    CUDA_CHECK(cudaMalloc((void**)&dev_accumulatedColor, getPixelCount() * sizeof(glm::vec3)));

    cudaMemset(dev_sampleCounts, 0, getPixelCount() * sizeof(unsigned int));

    cudaMemset(dev_accumulatedColor, 0, getPixelCount() * sizeof(glm::vec3));

    if (m_renderSettings.bSVGFEnabled) {
        m_svgfManager.resize(m_extent.width, m_extent.height);
    }
}

void Renderer::render(const std::unique_ptr<Scene>& scene, const std::unique_ptr<EnvironmentMap>& environmentMap, const Camera& camera, bool bClearAccumulatedSamples)
{
    if (m_cudaSurfaceObject == 0) {
        return;
    }

    if (bClearAccumulatedSamples) {
        cudaMemset(dev_sampleCounts, 0, getPixelCount() * sizeof(unsigned int));
        cudaMemset(dev_accumulatedColor, 0.0f, getPixelCount() * sizeof(glm::vec3));
        m_frameIndex = 0;
    }

    if (hasReachedSampleTarget()) {
        // Redraw the image so it makes it into the swapchain
        if (m_renderSettings.bSVGFEnabled) {
            m_svgfManager.display(m_cudaSurfaceObject);
        }
        else {
            launchDisplayAccumulatedSamplesKernel(m_cudaSurfaceObject,
                dev_accumulatedColor,
                dev_sampleCounts,
                m_extent.width,
                m_extent.height);
        }
         
        cudaDeviceSynchronize();
        return;
    }

    m_profiler.beginFrame();

    int initialActivePathCount = getPixelCount();
    int currentActivePathCount = initialActivePathCount;
    int maxBounces = m_renderSettings.maxBounces;

    {
        auto stage = m_profiler.scope("Camera Rays");
        launchCameraRayGenKernel(
            dev_pathStates,
            m_extent.width,
            m_extent.height,
            camera,
            m_frameIndex,
            m_renderSettings.bMSAAEnabled);
    }

    for (int i = 0; i < maxBounces; i++) {
        if (currentActivePathCount <= 0) {
            break;
        }

        {
            auto stage = m_profiler.scope("Intersect");
            launchIntersectKernel(dev_pathStates,
                dev_intersectionData,
                scene,
                currentActivePathCount);
        }

        if (i == 0 && m_renderSettings.bShowBvhHeatmap) {
            {
                auto stage = m_profiler.scope("BVH Heatmap");
                launchBvhHeatmapKernel(dev_pathStates,
                    dev_intersectionData,
                    currentActivePathCount,
                    m_cudaSurfaceObject,
                    m_extent.width,
                    m_renderSettings.bvhHeatmapMode,
                    m_renderSettings.bvhHeatmapMaxSteps);
            }

            cudaDeviceSynchronize();
            m_profiler.endFrame();
            return;
        }

        if (i == 0 && m_renderSettings.bSVGFEnabled) {
            auto stage = m_profiler.scope("SVGF G-Buffer");
            m_svgfManager.captureGBuffer(dev_pathStates, dev_intersectionData, currentActivePathCount, camera, scene);
        }

        if (m_renderSettings.bSortPathsByMaterial) {
            auto stage = m_profiler.scope("Material Sort");
            sortPathsByMaterial(dev_pathStates, dev_intersectionData, currentActivePathCount, scene->dev_materials);
        }

        {
            auto stage = m_profiler.scope("Shade");
            launchShadeKernel(dev_pathStates,
                dev_intersectionData,
                scene,
                environmentMap,
                currentActivePathCount,
                dev_accumulatedColor,
                m_renderSettings.bSVGFEnabled ? m_svgfManager.directChannel.dev_pingBuffer : nullptr,
                m_renderSettings.bSVGFEnabled ? m_svgfManager.indirectChannel.dev_pingBuffer : nullptr,
                dev_sampleCounts,
                m_renderSettings.lightSamplingMode,
                m_renderSettings.minRussianRouletteSurvival,
                i,
                m_frameIndex);
        }

        if (m_renderSettings.bStreamCompactionEnabled) {
            auto stage = m_profiler.scope("Stream Compaction");
            currentActivePathCount = runStreamCompaction(dev_pathStates, currentActivePathCount);
        }
    }

    if (currentActivePathCount > 0) {
        auto stage = m_profiler.scope("Record Active Paths");
        launchRecordActivePathsKernel(dev_pathStates,
            currentActivePathCount,
            dev_accumulatedColor,
            m_renderSettings.bSVGFEnabled ? m_svgfManager.directChannel.dev_pingBuffer : nullptr,
            m_renderSettings.bSVGFEnabled ? m_svgfManager.indirectChannel.dev_pingBuffer : nullptr,
            dev_sampleCounts);
    }

    // Draw to screen either through SVGF or our accumulated color buffer
    if (m_renderSettings.bSVGFEnabled) {
        auto stage = m_profiler.scope("SVGF Denoise");

        m_svgfManager.evaluate();

        m_svgfManager.display(m_cudaSurfaceObject);

        m_svgfManager.swapBuffers();
    }
    else {
        auto stage = m_profiler.scope("Display");
        launchDisplayAccumulatedSamplesKernel(m_cudaSurfaceObject,
            dev_accumulatedColor,
            dev_sampleCounts,
            m_extent.width,
            m_extent.height);
    }

    m_frameIndex++;

    cudaDeviceSynchronize();
    m_profiler.endFrame();
}

void Renderer::saveCurrentRenderToFile(const std::string& filepath)
{
    int numPixels = getPixelCount();
    if (numPixels <= 0) {
        return;
    }

    // Get the hdr color from the current output
    std::vector<glm::vec3> hdrColors(numPixels, glm::vec3(0.0f));

    if (m_renderSettings.bSVGFEnabled) {
        std::vector<glm::vec4> cpuFilteredColor(numPixels);
        CUDA_CHECK(cudaMemcpy(cpuFilteredColor.data(), m_svgfManager.dev_outputColor, numPixels * sizeof(glm::vec4), cudaMemcpyDeviceToHost));

        for (int i = 0; i < numPixels; i++) {
            hdrColors[i] = glm::vec3(cpuFilteredColor[i]);
        }
    }
    else {
        std::vector<glm::vec3> cpuAccumulatedColor(numPixels);
        std::vector<unsigned int> cpuSampleCounts(numPixels);

        CUDA_CHECK(cudaMemcpy(cpuAccumulatedColor.data(), dev_accumulatedColor, numPixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(cpuSampleCounts.data(), dev_sampleCounts, numPixels * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < numPixels; i++) {
            unsigned int samples = cpuSampleCounts[i];
            if (samples > 0) {
                hdrColors[i] = cpuAccumulatedColor[i] / static_cast<float>(samples);
            }
        }
    }

    // Convert to format readable by stb image
    std::vector<uint8_t> outputImage(numPixels * 3);

    for (int i = 0; i < numPixels; i++) {
        // Tonemap + gamma correct like we do in the cuda kernels
        glm::vec3 color = Tonemapping::toDisplayColor(hdrColors[i]);

        // Write into output image buffer
        outputImage[i * 3 + 0] = static_cast<uint8_t>(color.r * 255.99f);
        outputImage[i * 3 + 1] = static_cast<uint8_t>(color.g * 255.99f);
        outputImage[i * 3 + 2] = static_cast<uint8_t>(color.b * 255.99f);
    }

    stbi_flip_vertically_on_write(false);

    stbi_write_png(filepath.c_str(), m_extent.width, m_extent.height, 3, outputImage.data(), m_extent.width * 3);
}

void Renderer::drawRenderSettingsImGui(Camera& camera)
{
    ImGui::Text("Render Settings:");

    // Sample count
    ImGui::Text("Samples Per Pixel: %d%s", m_frameIndex, hasReachedSampleTarget() ? " (Done)" : "");
    ImGui::Checkbox("Stop at Target", &m_renderSettings.bLimitSamples);
    ImGui::SameLine();
    ImGui::BeginDisabled(!m_renderSettings.bLimitSamples);
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputInt("##TargetSamples", &m_renderSettings.targetSamplesPerPixel, 64, 1024)) {
        m_renderSettings.targetSamplesPerPixel = glm::max(m_renderSettings.targetSamplesPerPixel, 1);
    }
    ImGui::EndDisabled();

    if (ImGui::Checkbox("Use MSAA", &m_renderSettings.bMSAAEnabled)) {
		// Don't allow MSAA to be used with SVGF
        if(m_renderSettings.bMSAAEnabled && m_renderSettings.bSVGFEnabled) {
            // Turn off SVGF and free its buffers, keeping its settings
            m_svgfManager.resize(0, 0);
            m_renderSettings.bSVGFEnabled = false;
		}
	}
    if (ImGui::Checkbox("Use SVGF:", &m_renderSettings.bSVGFEnabled)) {
        if(m_renderSettings.bSVGFEnabled) {
            m_svgfManager.resize(m_extent.width, m_extent.height);
            // Don't allow MSAA to be used with SVGF
			m_renderSettings.bMSAAEnabled = false;
            // Reset lens radius to 0 to disable DOF
            camera.m_lensRadius = 0.0f;
            camera.m_hasChanged = true;
        }
        else {
            // Free SVGF buffers, keeping its settings
            m_svgfManager.resize(0, 0);
		}
    }

    // Sampling settings
    const char* lightSamplingLabels[] = { "BRDF Only", "NEE Only", "MIS" };
    int lightSamplingMode = static_cast<int>(m_renderSettings.lightSamplingMode);
    if (ImGui::Combo("Light Sampling", &lightSamplingMode, lightSamplingLabels, IM_ARRAYSIZE(lightSamplingLabels))) {
        m_renderSettings.lightSamplingMode = static_cast<LightSamplingMode>(lightSamplingMode);
        camera.m_hasChanged = true;
    }

    // Performance settings
    ImGui::Text("GPU Timings:");
    m_profiler.drawImGui();
    if (ImGui::SliderInt("Max Bounces", &m_renderSettings.maxBounces, 1, 32)) {
        camera.m_hasChanged = true;
    }
    if (ImGui::SliderFloat("Min RR Survival", &m_renderSettings.minRussianRouletteSurvival, 0.01f, 1.0f)) {
        camera.m_hasChanged = true;
    }
    ImGui::SetItemTooltip("Lowest chance a path survives russian roulette. 1 disables russian roulette");
    ImGui::Checkbox("Use Stream Compaction:", &m_renderSettings.bStreamCompactionEnabled);
	ImGui::Checkbox("Sort Paths by Material:", &m_renderSettings.bSortPathsByMaterial);

    // BVH settings
    ImGui::Text("BVH Heatmap:");
    if (ImGui::Checkbox("Show BVH Heatmap", &m_renderSettings.bShowBvhHeatmap)) {
        camera.m_hasChanged = true;
    }
    ImGui::BeginDisabled(!m_renderSettings.bShowBvhHeatmap);
    const char* heatmapModeLabels[] = { "Total", "BLAS", "TLAS" };
    int heatmapMode = static_cast<int>(m_renderSettings.bvhHeatmapMode);
    if (ImGui::Combo("Heatmap Mode", &heatmapMode, heatmapModeLabels, IM_ARRAYSIZE(heatmapModeLabels))) {
        m_renderSettings.bvhHeatmapMode = static_cast<BvhHeatmapMode>(heatmapMode);
    }
    ImGui::SliderInt("Max Steps", &m_renderSettings.bvhHeatmapMaxSteps, 1, 500);
    ImGui::EndDisabled();

	// SVGF settings
    ImGui::Text("SVGF Settings:");
    ImGui::BeginDisabled(!m_renderSettings.bSVGFEnabled);
    m_svgfManager.drawSettingsImGui();
    ImGui::EndDisabled();

	// Camera settings
    ImGui::Text("Camera Settings:");
    ImGui::BeginDisabled(m_renderSettings.bSVGFEnabled);
    if (ImGui::SliderFloat("Lens Radius", &camera.m_lensRadius, 0.0f, 0.25f)) {
        camera.m_hasChanged = true;
    }
    if (ImGui::SliderFloat("Focal Distance", &camera.m_focalDistance, 0.2, 100.f)) {
        camera.m_hasChanged = true;
    }
    ImGui::EndDisabled();
}

void Renderer::cleanup()
{
    if (m_cudaSurfaceObject) {
        cudaDestroySurfaceObject(m_cudaSurfaceObject);
        m_cudaSurfaceObject = 0;
    }
    if (m_cudaMipmappedArray) {
        cudaFreeMipmappedArray(m_cudaMipmappedArray);
        m_cudaMipmappedArray = nullptr;
    }
    if (m_cudaExtMemory) {
        cudaDestroyExternalMemory(m_cudaExtMemory);
        m_cudaExtMemory = nullptr;
    }
    m_cudaArray = nullptr;

    if (dev_pathStates) {
        cudaFree(dev_pathStates);
        dev_pathStates = nullptr;
    }

    if (dev_intersectionData) {
        cudaFree(dev_intersectionData);
        dev_intersectionData = nullptr;
    }

    if (dev_sampleCounts) {
        cudaFree(dev_sampleCounts);
        dev_sampleCounts = nullptr;
    }

    if (dev_accumulatedColor) {
        cudaFree(dev_accumulatedColor);
        dev_accumulatedColor = nullptr;
    }
}