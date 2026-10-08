#pragma once

// Debug views for SVGF
enum class SVGFView : int {
	Final = 0,
	Direct = 1, // Filtered direct lighting, remodulated with albedo
	Indirect = 2, // Filtered indirect lighting, remodulated with albedo
	Albedo = 3, // First hit albedo
	DirectVariance = 4,
	IndirectVariance = 5,
	HistoryLength = 6, // Frames of temporal history per pixel
	Normals = 7,
	MotionVectors = 8
};

struct SVGFSettings {
	SVGFView view = SVGFView::Final;

	// Temporal accumulation
	float colorAlpha = 0.2f;
	float momentsAlpha = 0.2f;

	// A-trous edge-stopping
	float sigmaLuminance = 4.0f;
	float sigmaNormal = 128.0f;
	float sigmaDepth = 1.0f;
	int atrousIterations = 5;
};
