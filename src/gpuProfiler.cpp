#include "gpuProfiler.h"

#include "cudaHelpers.h"

#include "imgui.h"

#include <algorithm>

// Scale for exponential moving average of frame times
static constexpr float SMOOTHING_FACTOR = 0.1f;

GpuProfiler::~GpuProfiler()
{
	for (cudaEvent_t event : m_eventPool) {
		cudaEventDestroy(event);
	}
}

void GpuProfiler::beginFrame()
{
	if (!m_bEnabled) {
		return;
	}

	m_bInFrame = true;
	m_eventsUsed = 0;
	m_records.clear();

	// Get an event for the start of the frame to capture total frame time
	m_frameStart = acquireEvent();
	CUDA_CHECK(cudaEventRecord(m_frameStart));
}

void GpuProfiler::endFrame()
{
	if (!m_bInFrame) {
		return;
	}
	m_bInFrame = false;
	
	// Get an event for the end of the frame to capture total frame time
	cudaEvent_t frameEnd = acquireEvent();
	CUDA_CHECK(cudaEventRecord(frameEnd));
	CUDA_CHECK(cudaEventSynchronize(frameEnd));

	// Reset each stage stats
	for (StageStats& stage : m_stages) {
		stage.frameMs = 0.0f;
		stage.bRanThisFrame = false;
	}

	// Add up the ms for each stage recorded this frame
	for (const StageRecord& record : m_records) {
		float elapsedMs = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, record.start, record.end));

		StageStats& stage = m_stages[record.stageIndex];
		stage.frameMs += elapsedMs;
		stage.bRanThisFrame = true;
	}
	
	// Calculate smoothed ms for each stage using exponential moving average
	for (StageStats& stage : m_stages) {
		stage.smoothedMs += SMOOTHING_FACTOR * (stage.frameMs - stage.smoothedMs);
	}
	
	// Calculate smoothed ms for the total frame using exponential moving average
	float frameMs = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&frameMs, m_frameStart, frameEnd));
	m_smoothedFrameMs += SMOOTHING_FACTOR * (frameMs - m_smoothedFrameMs);
}

void GpuProfiler::beginStage(const char* name)
{
	if (!m_bInFrame) {
		return;
	}

	// Get a stage, checking if one with the same name already exists (in which case we share the same index)
	m_openStageIndex = findOrAddStage(name);
	m_openStageStart = acquireEvent();
	CUDA_CHECK(cudaEventRecord(m_openStageStart));
}

void GpuProfiler::endStage()
{
	if (!m_bInFrame || m_openStageIndex < 0) {
		return;
	}
	
	// Get an event for the end of the stage to capture its duration
	cudaEvent_t stageEnd = acquireEvent();
	CUDA_CHECK(cudaEventRecord(stageEnd));

	m_records.push_back({ m_openStageIndex, m_openStageStart, stageEnd });
	m_openStageIndex = -1;
}

void GpuProfiler::drawImGui()
{
	ImGui::Checkbox("Profile GPU Stages", &m_bEnabled);
	if (!m_bEnabled) {
		return;
	}

	// Make a table of stage name, ms, and percent of the total frame time
	if (!ImGui::BeginTable("GpuStages", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
		return;
	}

	ImGui::TableSetupColumn("Stage");
	ImGui::TableSetupColumn("ms");
	ImGui::TableSetupColumn("%");
	ImGui::TableHeadersRow();

	auto drawRow = [this](const char* name, float ms) {
		float percent = m_smoothedFrameMs > 0.0f ? 100.0f * ms / m_smoothedFrameMs : 0.0f;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::TextUnformatted(name);
		ImGui::TableNextColumn();
		ImGui::Text("%.2f", ms);
		ImGui::TableNextColumn();
		ImGui::Text("%.0f%%", percent);
	};

	float trackedMs = 0.0f;
	for (const StageStats& stage : m_stages) {
		// Hide stages that are turned off during this frame
		if (!stage.bRanThisFrame) {
			continue;
		}
		drawRow(stage.name.c_str(), stage.smoothedMs);
		trackedMs += stage.smoothedMs;
	}

	// Draw a row for any time not accounted for by the tracked stages
	drawRow("Untracked", (std::max)(0.0f, m_smoothedFrameMs - trackedMs));

	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted("GPU Frame");
	ImGui::TableNextColumn();
	ImGui::Text("%.2f", m_smoothedFrameMs);

	ImGui::EndTable();
}

cudaEvent_t GpuProfiler::acquireEvent()
{
	if (m_eventsUsed == m_eventPool.size()) {
		cudaEvent_t event = nullptr;
		CUDA_CHECK(cudaEventCreate(&event));
		m_eventPool.push_back(event);
	}

	return m_eventPool[m_eventsUsed++];
}

int GpuProfiler::findOrAddStage(const char* name)
{
	for (size_t i = 0; i < m_stages.size(); i++) {
		if (m_stages[i].name == name) {
			return static_cast<int>(i);
		}
	}

	StageStats stage;
	stage.name = name;
	m_stages.push_back(stage);
	return static_cast<int>(m_stages.size()) - 1;
}
