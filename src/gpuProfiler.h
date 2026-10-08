#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

// Times GPU work per named stage using CUDA events on the default stream.
// Stages with the same name in one frame (like per bounce kernels) are summed together,
// and timings are smoothed over frames so they're readable in ImGui.
class GpuProfiler {
public:
	// When this goes out of scope, it will record the time for the stage and be added to the profiler
	class ScopedStage {
	public:
		ScopedStage(GpuProfiler& profiler, const char* name) : m_profiler(profiler) {
			m_profiler.beginStage(name);
		}
		~ScopedStage() {
			m_profiler.endStage();
		}

		ScopedStage(const ScopedStage&) = delete;
		ScopedStage& operator=(const ScopedStage&) = delete;

	private:
		GpuProfiler& m_profiler;
	};

	GpuProfiler() = default;
	~GpuProfiler();

	GpuProfiler(const GpuProfiler&) = delete;
	GpuProfiler& operator=(const GpuProfiler&) = delete;

	void beginFrame();

	void endFrame();

	// No support for nesting stages currently
	void beginStage(const char* name);
	void endStage();

	ScopedStage scope(const char* name) {
		return ScopedStage(*this, name);
	}

	void drawImGui();

private:
	struct StageStats {
		std::string name;
		float frameMs = 0.0f;
		float smoothedMs = 0.0f;
		bool bRanThisFrame = false;
	};

	struct StageRecord {
		int stageIndex;
		cudaEvent_t start;
		cudaEvent_t end;
	};

	cudaEvent_t acquireEvent();
	int findOrAddStage(const char* name);

	bool m_bEnabled = true;
	bool m_bInFrame = false;

	// Kept in first seen order
	std::vector<StageStats> m_stages;

	std::vector<StageRecord> m_records;
	int m_openStageIndex = -1;
	cudaEvent_t m_openStageStart = nullptr;

	// Cuda events are reused every frame so there is less impact on perf
	std::vector<cudaEvent_t> m_eventPool;
	size_t m_eventsUsed = 0;

	cudaEvent_t m_frameStart = nullptr;
	float m_smoothedFrameMs = 0.0f;
};
