#pragma once

enum class BvhHeatmapMode : int {
	Total = 0,
	Blas = 1,
	Tlas = 2
};

// Mode for light sampling so we can test different strategies
enum class LightSamplingMode : int {
	BrdfOnly = 0,
	NeeOnly = 1,
	Mis = 2
};

struct RenderSettings {
	bool bMSAAEnabled = true;
	bool bSVGFEnabled = false;
	int maxBounces = 6;
	LightSamplingMode lightSamplingMode = LightSamplingMode::Mis;

	// When set to 1 it disables russian roulette
	float minRussianRouletteSurvival = 0.05f;

	// Stop rendering once every pixel has this many samples, for equal sample comparisons
	bool bLimitSamples = false;
	int targetSamplesPerPixel = 1024;

	bool bStreamCompactionEnabled = false;
	bool bSortPathsByMaterial = false;

	bool bShowBvhHeatmap = false;
	BvhHeatmapMode bvhHeatmapMode = BvhHeatmapMode::Total;
	int bvhHeatmapMaxSteps = 100; // Limit number of steps so anything greater is red
};
