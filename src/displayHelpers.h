#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

// Writes a [0, 1] color to the surface in BGRA8 format
__device__ inline void writeSurfacePixel(cudaSurfaceObject_t surface, int x, int y, const glm::vec3& color) {
	glm::vec3 clamped = glm::clamp(color, 0.0f, 1.0f);

	uchar4 pixelColor;
	pixelColor.x = (unsigned char)(clamped.b * 255.0f);
	pixelColor.y = (unsigned char)(clamped.g * 255.0f);
	pixelColor.z = (unsigned char)(clamped.r * 255.0f);
	pixelColor.w = 255;

	surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

// Maps t from 0 -> 1 to a range of colors from blue -> red
__device__ inline glm::vec3 heatmapColor(float t) {
	t = glm::clamp(t, 0.0f, 1.0f);

	const glm::vec3 stops[5] = {
		glm::vec3(0.0f, 0.0f, 1.0f), // Blue
		glm::vec3(0.0f, 1.0f, 1.0f), // Cyan
		glm::vec3(0.0f, 1.0f, 0.0f), // Green
		glm::vec3(1.0f, 1.0f, 0.0f), // Yellow
		glm::vec3(1.0f, 0.0f, 0.0f)  // Red
	};

	float scaled = t * 4.0f;
	int lower = glm::min((int)scaled, 3);
	return glm::mix(stops[lower], stops[lower + 1], scaled - (float)lower);
}
