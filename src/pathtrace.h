#pragma once

#include "scene.h"
#include "utilities.h"

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

inline constexpr std::uint32_t kRendererSamplerVersion = 2;

enum class RendererDebugMode : std::uint32_t
{
    None = 0,
    Normals = 1,
    Uv = 2,
    Depth = 3
};

struct RendererConfig
{
    std::uint64_t seed = 1;
    int maxDepth = 8;
    bool aa = true;
    bool nee = true;
    bool useBvh = true;
    bool compact = true;
    bool sort = false;
    bool benchmark = false;
    RendererDebugMode debug = RendererDebugMode::None;
};

struct RendererBenchmarkRow
{
    std::uint64_t sample = 0;
    int bounce = 0;
    int activeBefore = 0;
    int activeAfter = 0;
    float intersectionMs = 0.0f;
    float sortMs = 0.0f;
    float shadeMs = 0.0f;
    float compactionMs = 0.0f;
    float countMs = 0.0f;
};

struct RendererBenchmarkSummary
{
    std::uint64_t measuredSamples = 0;
    double hostMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
};

struct RendererValidationCheck
{
    std::string name;
    bool passed = false;
    std::string detail;
    // Optional renderer-authored JSON object emitted as structured check data.
    std::string dataJson;
};

struct RendererValidationReport
{
    bool passed = false;
    std::vector<RendererValidationCheck> checks;
};

struct RendererCompatibilityInfo
{
    std::string sourceBuild;
    std::string deviceName;
    int computeMajor = 0;
    int computeMinor = 0;
    int runtimeVersion = 0;
    int driverVersion = 0;
};

// Starter-compatible preview entry points.
void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int frame, int iteration);

// Explicit renderer control used by headless/checkpoint/validation paths.
void pathtraceSetConfig(const RendererConfig& config);
RendererConfig pathtraceGetConfig();
void pathtraceResetAccumulation();
void pathtraceRunSample();
std::uint64_t pathtraceSampleCount();
void pathtraceDownloadAccumulation(std::vector<glm::vec3>& accumulation);
void pathtraceUploadAccumulation(const std::vector<glm::vec3>& accumulation,
                                 std::uint64_t completedSamples);
std::vector<RendererBenchmarkRow> pathtraceTakeBenchmarkRows();
RendererBenchmarkSummary pathtraceBenchmarkSummary();
RendererValidationReport pathtraceValidate();
std::string pathtraceValidationJson(const RendererValidationReport& report);
RendererCompatibilityInfo pathtraceCompatibilityInfo();
