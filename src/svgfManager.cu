#include "svgfManager.h"
#include "cudaHelpers.h"

#include "intersection.h"
#include "material.h"
#include "tonemapping.h"
#include "displayHelpers.h"

#include <cuda_runtime.h>

#include "imgui.h"

#include <initializer_list>

constexpr unsigned int MAX_HISTORY_LENGTH = 32;

__device__ glm::vec2 calculateMotionVector(
	glm::vec3 worldPos,
	glm::mat4 currentViewProj,
	glm::mat4 prevViewProj,
	int width,
	int height) 
{
	glm::vec4 currentClip = currentViewProj * glm::vec4(worldPos, 1.0f);
	glm::vec4 prevClip = prevViewProj * glm::vec4(worldPos, 1.0f);

	if (currentClip.w == 0.0f || prevClip.w == 0.0f) {
		return glm::vec2(0.0f);
	}

	glm::vec2 currentNDC = glm::vec2(currentClip.x / currentClip.w, currentClip.y / currentClip.w);
	glm::vec2 prevNDC = glm::vec2(prevClip.x / prevClip.w, prevClip.y / prevClip.w);

	glm::vec2 currentScreen = glm::vec2(
		(currentNDC.x + 1.0f) * 0.5f * width,
		(1.0f - currentNDC.y) * 0.5f * height
	);

	glm::vec2 prevScreen = glm::vec2(
		(prevNDC.x + 1.0f) * 0.5f * width,
		(1.0f - prevNDC.y) * 0.5f * height
	);

	return prevScreen - currentScreen;
}

__device__ float computeLuminance(const glm::vec3& color)
{
	glm::vec3 luminanceWeights = glm::vec3(0.2126f, 0.7152f, 0.0722f);
	float luminance = glm::dot(color, luminanceWeights);
	return luminance;
}

__global__ void kernCaptureGBuffer(
	const PathState* dev_pathStates,
	const IntersectionData* dev_intersectionData,
	glm::vec4* dev_gBuffer_normalDepth,
	glm::vec2* dev_gBuffer_motionVectors,
	glm::vec3* dev_gBuffer_albedo,
	int* dev_gBuffer_geometryId,
	glm::mat4 currentViewProj,
	glm::mat4 prevViewProj,
	int width,
	int height,
	int currentActivePathCount,
	DevScene dev_scene)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= currentActivePathCount) {
		return;
	}

	const PathState& pathState = dev_pathStates[index];
	const IntersectionData& intersectionData = dev_intersectionData[index];

	int pixelIdx = pathState.pixelIndex;

	glm::vec3 normal = intersectionData.normal;
	glm::vec3 worldPos = pathState.ray.getPositionAtTime(intersectionData.t);
	float depth = intersectionData.t;

	if (depth < 0.0f || depth >= 1e10f) {
		dev_gBuffer_normalDepth[pixelIdx] = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
		dev_gBuffer_motionVectors[pixelIdx] = glm::vec2(0.0f);
		dev_gBuffer_albedo[pixelIdx] = glm::vec3(1.0f); // Don't demodulate a miss 
		dev_gBuffer_geometryId[pixelIdx] = -1;
		return;
	}

	dev_gBuffer_normalDepth[pixelIdx] = glm::vec4(normal, depth);
	dev_gBuffer_geometryId[pixelIdx] = intersectionData.geometryIndex;

	glm::vec2 motionVector = calculateMotionVector(worldPos, currentViewProj, prevViewProj, width, height);
	dev_gBuffer_motionVectors[pixelIdx] = motionVector;

	// Demodulation albedo. Emitters and delta materials keep 1 since what they show isn't their own surface texture
	glm::vec3 albedo = glm::vec3(1.0f);
	if (intersectionData.materialIndex >= 0 && (size_t)intersectionData.materialIndex < dev_scene.m_materialCount) {
		Material material = dev_scene.dev_materials[intersectionData.materialIndex];
		material.initializeFromIntersection(intersectionData);

		if (material.emittance <= 0.0f && !material.isDelta()) {
			// Only demodulate non-emissive, non-delta materials
			albedo = material.getDemodulationAlbedo();
		}
	}

	// Clamp albedo to a minimum to avoid divide by zero
	const float minDemodulationAlbedo = 0.01f;
	dev_gBuffer_albedo[pixelIdx] = glm::max(albedo, glm::vec3(minDemodulationAlbedo));
}
__device__ bool isHistoryValid(
	glm::vec3 currNormal, float currDepth, int currGeometryId,
	glm::vec3 prevNormal, float prevDepth, int prevGeometryId
)
{
	// Don't let history bleed across different objects
	if (currGeometryId != prevGeometryId) {
		return false;
	}

	float normalSimilarity = glm::dot(currNormal, prevNormal);
	float depthDifference = glm::abs(currDepth - prevDepth) / glm::max(currDepth, 0.0001f);

	return normalSimilarity > 0.95f && depthDifference < 0.1f && prevDepth >= 0.0f;
}

__global__ void kernTemporalAccumulation(
	const glm::vec4* dev_gBuffer_normalDepth,
	const glm::vec2* dev_gBuffer_motionVectors,
	const glm::vec4* dev_gBuffer_normalDepthPrev,
	const int* dev_gBuffer_geometryId,
	const int* dev_gBuffer_geometryIdPrev,
	const glm::vec4* dev_pingBuffer,      
	const glm::vec4* dev_illuminationPrev,
	const glm::vec2* dev_momentsPrev,

	// Output Buffers
	glm::vec4* dev_integratedColor,
	glm::vec2* dev_moments,
	unsigned int* dev_historyLength,
	const unsigned int* dev_historyLengthPrev,
	float colorAlphaMin,
	float momentsAlphaMin,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;

	glm::vec4 currentND = dev_gBuffer_normalDepth[pixelIndex];
	float currentDepth = currentND.w;

	if (currentDepth < 0.0f) {
		dev_integratedColor[pixelIndex] = dev_pingBuffer[pixelIndex];
		dev_moments[pixelIndex] = glm::vec2(0.0f);
		dev_historyLength[pixelIndex] = 0;
		return;
	}

	glm::vec3 currentNormal = glm::vec3(currentND.x, currentND.y, currentND.z);
	glm::vec4 currentColor = dev_pingBuffer[pixelIndex];
	float currentLuminance = glm::clamp(computeLuminance(currentColor), 0.0f, 30.0f);
	glm::vec2 currentMoments = glm::vec2(currentLuminance, currentLuminance * currentLuminance);

	// Get prev x and y from motion vector
	glm::vec2 motionVector = dev_gBuffer_motionVectors[pixelIndex];
	glm::vec2 prevSubpixel = glm::vec2(x, y) + motionVector;

	// Get a 2x2 window
	int x0 = __float2int_rd(prevSubpixel.x);
	int y0 = __float2int_rd(prevSubpixel.y);
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	glm::vec2 a = prevSubpixel - glm::vec2(x0, y0);

	// Clamp to edges
	x0 = glm::clamp(x0, 0, width - 1);
	x1 = glm::clamp(x1, 0, width - 1);
	y0 = glm::clamp(y0, 0, height - 1);
	y1 = glm::clamp(y1, 0, height - 1);

	// Get neighborhood values
	int idx00 = y0 * width + x0;
	int idx10 = y0 * width + x1;
	int idx01 = y1 * width + x0;
	int idx11 = y1 * width + x1;

	glm::vec4 nd00 = dev_gBuffer_normalDepthPrev[idx00];
	glm::vec4 nd10 = dev_gBuffer_normalDepthPrev[idx10];
	glm::vec4 nd01 = dev_gBuffer_normalDepthPrev[idx01];
	glm::vec4 nd11 = dev_gBuffer_normalDepthPrev[idx11];

	// Check neighborhood validity
	int currentGeometryId = dev_gBuffer_geometryId[pixelIndex];
	bool v00 = isHistoryValid(currentNormal, currentDepth, currentGeometryId, glm::vec3(nd00), nd00.w, dev_gBuffer_geometryIdPrev[idx00]);
	bool v10 = isHistoryValid(currentNormal, currentDepth, currentGeometryId, glm::vec3(nd10), nd10.w, dev_gBuffer_geometryIdPrev[idx10]);
	bool v01 = isHistoryValid(currentNormal, currentDepth, currentGeometryId, glm::vec3(nd01), nd01.w, dev_gBuffer_geometryIdPrev[idx01]);
	bool v11 = isHistoryValid(currentNormal, currentDepth, currentGeometryId, glm::vec3(nd11), nd11.w, dev_gBuffer_geometryIdPrev[idx11]);

	// Bilinear blending coefficients
	float w00 = (1.0f - a.x) * (1.0f - a.y);
	float w10 = a.x * (1.0f - a.y);
	float w01 = (1.0f - a.x) * a.y;
	float w11 = a.x * a.y;

	glm::vec4 accumulatedColor = glm::vec4(0.0f);
	glm::vec2 accumulatedMoments = glm::vec2(0.0f);
	float totalHistoryCount = 0.0f;
	float totalBilinearWeight = 0.0f;

	// Run bilinear filtering
	if (v00) {
		accumulatedColor += dev_illuminationPrev[idx00] * w00; 
		accumulatedMoments += dev_momentsPrev[idx00] * w00; 
		totalHistoryCount += (float)dev_historyLengthPrev[idx00] * w00; 
		totalBilinearWeight += w00;
	}
	if (v10) { 
		accumulatedColor += dev_illuminationPrev[idx10] * w10; 
		accumulatedMoments += dev_momentsPrev[idx10] * w10; 
		totalHistoryCount += (float)dev_historyLengthPrev[idx10] * w10; 
		totalBilinearWeight += w10; 
	}
	if (v01) { 
		accumulatedColor += dev_illuminationPrev[idx01] * w01; 
		accumulatedMoments += dev_momentsPrev[idx01] * w01; 
		totalHistoryCount += (float)dev_historyLengthPrev[idx01] * w01; 
		totalBilinearWeight += w01; 
	}
	if (v11) { 
		accumulatedColor += dev_illuminationPrev[idx11] * w11; 
		accumulatedMoments += dev_momentsPrev[idx11] * w11; 
		totalHistoryCount += (float)dev_historyLengthPrev[idx11] * w11; 
		totalBilinearWeight += w11; 
	}

	if (!MathHelpers::isFinite(glm::vec3(accumulatedColor)) || !MathHelpers::isFinite(accumulatedMoments)) {
		// Sometimes these can be infinte or nan so if that is the case, set the weight to 0 so it can be reset to 1 spp
		totalBilinearWeight = 0.0f;
	}

	if (totalBilinearWeight > 0.0f) {
		// Normalize attributes
		accumulatedColor /= totalBilinearWeight;
		accumulatedMoments /= totalBilinearWeight;

		// Update history length
		unsigned int historicalCount = __float2uint_rn(totalHistoryCount / totalBilinearWeight);
		unsigned int updatedHistoryCount = glm::min(historicalCount + 1, MAX_HISTORY_LENGTH);
		dev_historyLength[pixelIndex] = updatedHistoryCount;

		float colorAlpha = glm::max(1.0f / (float)updatedHistoryCount, colorAlphaMin);
		float momentsAlpha = glm::max(1.0f / (float)updatedHistoryCount, momentsAlphaMin);

		// Blend color and moments
		dev_integratedColor[pixelIndex] = glm::mix(accumulatedColor, currentColor, colorAlpha);
		dev_moments[pixelIndex] = glm::mix(accumulatedMoments, currentMoments, momentsAlpha);
	}
	else {
		// Failed to remap, reset to 1 spp
		dev_historyLength[pixelIndex] = 1;
		dev_integratedColor[pixelIndex] = currentColor;
		dev_moments[pixelIndex] = currentMoments;
	}
}

__device__ float calculateVariance(const glm::vec2& moments) {
	return glm::max(0.0f, moments.y - moments.x * moments.x);
}

__global__ void kernEstimateVariance(
	const glm::vec4* dev_pingBuffer,
	const glm::vec2* dev_moments,
	const glm::vec4* dev_gBuffer_normalDepth,
	const unsigned int* dev_historyLength,
	float* dev_variance,
	int width, int height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;
	unsigned int history = dev_historyLength[pixelIndex];

	// Need some significant history to calculate variance
	if (history >= 4) {
		glm::vec2 m = dev_moments[pixelIndex];

		dev_variance[pixelIndex] = calculateVariance(m);
		return;
	}
	dev_variance[pixelIndex] = 0.0f;

	// If short history, use spatial varaince
	glm::vec4 centerND = dev_gBuffer_normalDepth[pixelIndex];
	float centerDepth = centerND.w;

	if (centerDepth < 0.0f) {
		dev_variance[pixelIndex] = 0.0f;
		return;
	}

	glm::vec3 centerNormal = glm::vec3(centerND);

	glm::vec2 sumMoments = glm::vec2(0.0f);
	float sumWeight = 0.0f;

	// 7x7 bilateral filter
	for (int dy = -3; dy <= 3; dy++) {
		for (int dx = -3; dx <= 3; dx++) {
			int nx = x + dx;
			int ny = y + dy;

			if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
				continue;
			}

			int neighborIdx = ny * width + nx;
			glm::vec4 neighborND = dev_gBuffer_normalDepth[neighborIdx];
			float neighborDepth = neighborND.w;

			if (neighborDepth < 0.0f) {
				continue;
			}

			glm::vec3 neighborNormal = glm::vec3(neighborND);

			// Bilateral weights
			float wNormal = glm::pow(glm::max(0.0f, glm::dot(centerNormal, neighborNormal)), 128.0f);
			float wDepth = glm::exp(-glm::abs(centerDepth - neighborDepth) / (centerDepth * 0.1f + 1e-4f));
			
			unsigned int neighborHistory = dev_historyLength[neighborIdx];
			float wHistory = glm::max(1.0f, (float)neighborHistory);

			float wBilateral = wNormal * wDepth * wHistory;

			glm::vec2 nMoments = dev_moments[neighborIdx];
			sumMoments += wBilateral * nMoments;
			sumWeight += wBilateral;
		}
	}

	if (sumWeight > 0.0f) {
		sumMoments /= sumWeight;
	}

	dev_variance[pixelIndex] = calculateVariance(sumMoments);
}

__global__ void kernVariancePrefilter3x3(
	const float* dev_variance,
	const glm::vec4* dev_gBuffer_normalDepth,
	float* dev_prefilteredVariance,
	int width, int height
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	float centerDepth = dev_gBuffer_normalDepth[pixelIndex].w;
	if (centerDepth < 0.0f) {
		// Skip computing variance
		dev_prefilteredVariance[pixelIndex] = 0.0f;
		return;
	}

	// 3x3 Gaussian kernel
	const float wGaus[3][3] = {
		{ 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f },
		{ 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f },
		{ 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f }
	};

	float sumVariance = 0.0f;
	float sumWeight = 0.0f;

	// Blur over 3x3 neighborhood
	for (int r = -1; r <= 1; ++r) {
		for (int c = -1; c <= 1; ++c) {
			int nx = glm::clamp(x + c, 0, width - 1);
			int ny = glm::clamp(y + r, 0, height - 1);
			int neighborIndex = ny * width + nx;

			float neighborDepth = dev_gBuffer_normalDepth[neighborIndex].w;
			if (neighborDepth < 0.0f) {
				// Neighbor missed geometry, skip
				continue;
			}

			float w = wGaus[r + 1][c + 1];
			sumVariance += dev_variance[neighborIndex] * w;
			sumWeight += w;
		}
	}

	if (sumWeight > 0.0f) {
		dev_prefilteredVariance[pixelIndex] = sumVariance / sumWeight;
	}
	else {
		// Copy non blurred variance for rays missing geometry
		dev_prefilteredVariance[pixelIndex] = dev_variance[pixelIndex];
		//dev_prefilteredVariance[pixelIndex] = 0.0f;
	}
}

__device__ glm::vec2 computeDepthGradient(
	glm::ivec2 p,
	const glm::vec4* dev_gBuffer_normalDepth,
	float depthP,
	int width,
	int height
) {
	int rightX = glm::min(p.x + 1, width - 1);
	int leftX = glm::max(p.x - 1, 0);
	int bottomY = glm::min(p.y + 1, height - 1);
	int topY = glm::max(p.y - 1, 0);

	float depthRight = dev_gBuffer_normalDepth[p.y * width + rightX].w;
	float depthLeft = dev_gBuffer_normalDepth[p.y * width + leftX].w;
	float depthBottom = dev_gBuffer_normalDepth[bottomY * width + p.x].w;
	float depthTop = dev_gBuffer_normalDepth[topY * width + p.x].w;

	float dDepthdX = 0.0f;
	if (depthRight >= 0.0f) {
		dDepthdX = depthRight - depthP;
	}
	else if (depthLeft >= 0.0f) {
		dDepthdX = depthP - depthLeft;
	}

	float dDepthdY = 0.0f;
	if (depthBottom >= 0.0f) {
		dDepthdY = depthBottom - depthP;
	}
	else if (depthTop >= 0.0f) {
		dDepthdY = depthP - depthTop;
	}

	return glm::vec2(dDepthdX, dDepthdY);
}

__device__ float computeTotalWeight(
	// Pixel values
	glm::ivec2 p,
	glm::ivec2 q,
	// Depth values
	float depthP,
	float depthQ,
	float sigmaDepth,
	glm::vec2 depthGradP,
	// Normal values
	glm::vec3 normalP,
	glm::vec3 normalQ,
	float sigmaNormal,
	// Luminance values
	float luminanceP,
	float luminanceQ,
	float luminancePrefilteredVariance,
	float sigmaLuminance
)
{
	const float epsilon = 0.0001f;
	// Calculate w_z
	float depthDelta = glm::abs(depthP - depthQ);
	float w_z = glm::exp(-depthDelta / (sigmaDepth * glm::abs(glm::dot(depthGradP, glm::vec2(p - q))) + epsilon));

	// Calculate w_n
	float normalDot = glm::dot(normalP, normalQ);
	float w_n = glm::pow(glm::max(0.0f, normalDot), sigmaNormal);

	// Calculate w_l
	float luminanceDelta = glm::abs(luminanceP - luminanceQ);
	float w_l = glm::exp(-luminanceDelta / (sigmaLuminance * glm::sqrt(luminancePrefilteredVariance) + epsilon));

	float w_i = w_z * w_n * w_l;

	return w_i;
}

__global__ void kernAtrousFilter(
	const glm::vec4* dev_inputColor,
	const glm::vec4* dev_gBuffer_normalDepth,
	const float* dev_inputVariance,
	const float* dev_prefilteredVariance,
	float* dev_outputVariance,
	glm::vec4* dev_outputColor,
	int stride,
	float sigmaLuminance,
	float sigmaNormal,
	float sigmaDepth,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;
	glm::ivec2 p = glm::ivec2(x, y);

	glm::vec4 centerND = dev_gBuffer_normalDepth[pixelIndex];
	float depthP = centerND.w;

	if (depthP < 0.0f) {
		// Ray missed, just copy without filtering
		dev_outputColor[pixelIndex] = dev_inputColor[pixelIndex];
		return;
	}

	glm::vec2 depthGradP = computeDepthGradient(p, dev_gBuffer_normalDepth, depthP, width, height);
	glm::vec3 normalP = glm::vec3(centerND);
	glm::vec4 colorP = dev_inputColor[pixelIndex];
	float luminanceP = computeLuminance(glm::vec3(colorP));
	float varianceP = dev_inputVariance[pixelIndex];
	float prefilteredVarianceP = dev_prefilteredVariance[pixelIndex];

	// h filter kernel 
	const float h[5] = { 1.0f/16.0f, 1.0f/4.0f, 3.0f/8.0f, 1.0f/4.0f, 1.0f/16.0f };

	glm::vec3 sumColor = glm::vec3(0.0f);
	float sumColorWeight = 0.0f;

	float sumVariance = 0.0f;
	float sumVarianceWeight = 0.0f;

	// Filter over 5x5 window
	for (int r = -2; r <= 2; ++r) {
		for (int c = -2; c <= 2; ++c) {
			if (r == 0 && c == 0) {
				float wKernel = h[2] * h[2];
				sumColor += glm::vec3(colorP) * wKernel;
				sumColorWeight += wKernel;
				sumVariance += wKernel * wKernel * varianceP;
				sumVarianceWeight += wKernel;
				continue;
			}

			// Calculate neighbor index
			int nx = glm::clamp(x + c * stride, 0, width - 1);
			int ny = glm::clamp(y + r * stride, 0, height - 1);
			glm::ivec2 q = glm::ivec2(nx, ny);
			int neighborIndex = ny * width + nx;



			glm::vec4 neighborND = dev_gBuffer_normalDepth[neighborIndex];
			float depthQ = neighborND.w;

			if (depthQ < 0.0f) {
				// Neighbor hit environment map, discard
				continue;
			}

			glm::vec3 normalQ = glm::vec3(neighborND);
			glm::vec4 colorQ = dev_inputColor[neighborIndex];
			float luminanceQ = computeLuminance(glm::vec3(colorQ));
			float varianceQ = dev_inputVariance[neighborIndex];

			float h_q = h[r + 2] * h[c + 2];

			float w_pq = computeTotalWeight(
				p, q,
				depthP, depthQ, sigmaDepth, depthGradP,
				normalP, normalQ, sigmaNormal,
				luminanceP, luminanceQ, prefilteredVarianceP, sigmaLuminance
			);

			sumColor += h_q * w_pq * glm::vec3(colorQ);
			sumColorWeight += h_q * w_pq;

			sumVariance += h_q * h_q * w_pq * w_pq * varianceQ;
			sumVarianceWeight += h_q * w_pq;
		}
	}

	if (sumColorWeight > 0.0f) {
		dev_outputColor[pixelIndex] = glm::vec4(sumColor / sumColorWeight, colorP.w);
	}
	else {
		dev_outputColor[pixelIndex] = colorP;
	}

	if (sumVarianceWeight > 0.0f) {
		dev_outputVariance[pixelIndex] = sumVariance / (sumVarianceWeight * sumVarianceWeight);
	}
	else {
		dev_outputVariance[pixelIndex] = varianceP;
	}
}

// Remap a vector from [-1, 1] to [0, 1] so it can be visualized
__device__ void drawVec3ToSurface(const glm::vec3& vector, cudaSurfaceObject_t surface, int x, int y) {
	writeSurfacePixel(surface, x, y, vector * 0.5f + 0.5f);
}

__global__ void kernDebugNormals(glm::vec4* dev_gBuffer_normalDepth, cudaSurfaceObject_t surface, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;
	const glm::vec4& normalDepth = dev_gBuffer_normalDepth[pixelIndex];

	drawVec3ToSurface(normalDepth, surface, x, y);
}

__global__ void kernDebugMotionVectors(glm::vec2* dev_gBuffer_motionVectors, cudaSurfaceObject_t surface, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;
	const glm::vec2& motionVector = dev_gBuffer_motionVectors[pixelIndex];

	glm::vec3 visualVector = glm::vec3(motionVector.x * 0.05f, motionVector.y * 0.05f, 0.0f);
	drawVec3ToSurface(visualVector, surface, x, y);
}

__global__ void kernDebugSVGFIllumination(
	const glm::vec4* dev_pingBuffer,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	glm::vec3 hdrColor = glm::vec3(dev_pingBuffer[pixelIndex]);

	// Tonemap for display purposes
	glm::vec3 finalColor = Tonemapping::toDisplayColor(hdrColor);

	writeSurfacePixel(surface, x, y, finalColor);
}

__global__ void kernDebugVariance(
	const float* dev_variance,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;
	float variance = dev_variance[pixelIndex];

	// Amplify variance visually since raw statistical variance values are tiny decimal numbers
	float visualIntensity = variance * 10.0f;

	writeSurfacePixel(surface, x, y, glm::vec3(visualIntensity, 0.0f, visualIntensity));
}

// Draws one filtered lighting channel, remodulated
__global__ void kernDisplayChannel(
	const glm::vec4* dev_channel,
	const glm::vec3* dev_gBuffer_albedo,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	glm::vec3 hdrColor = glm::vec3(dev_channel[pixelIndex]) * dev_gBuffer_albedo[pixelIndex];
	writeSurfacePixel(surface, x, y, Tonemapping::toDisplayColor(hdrColor));
}

__global__ void kernDisplayAlbedo(
	const glm::vec3* dev_gBuffer_albedo,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	// Albedo is already in [0, 1], so just gamma correct to display
	writeSurfacePixel(surface, x, y, Tonemapping::gammaCorrect(dev_gBuffer_albedo[pixelIndex], 2.2f));
}

__global__ void kernDisplayHistoryLength(
	const unsigned int* dev_historyLength,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	// Blue for no history, red for a full history
	float t = (float)dev_historyLength[pixelIndex] / (float)MAX_HISTORY_LENGTH;
	writeSurfacePixel(surface, x, y, heatmapColor(t));
}

__global__ void kernDemodulateAlbedo(
	glm::vec4* dev_direct,
	glm::vec4* dev_indirect,
	const glm::vec3* dev_gBuffer_albedo,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	glm::vec3 albedo = dev_gBuffer_albedo[pixelIndex];
	glm::vec4 direct = dev_direct[pixelIndex];
	glm::vec4 indirect = dev_indirect[pixelIndex];

	dev_direct[pixelIndex] = glm::vec4(glm::vec3(direct) / albedo, direct.w);
	dev_indirect[pixelIndex] = glm::vec4(glm::vec3(indirect) / albedo, indirect.w);
}

__global__ void kernCombineChannels(
	const glm::vec4* dev_direct,
	const glm::vec4* dev_indirect,
	const glm::vec3* dev_gBuffer_albedo,
	glm::vec4* dev_outputColor,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	glm::vec4 direct = dev_direct[pixelIndex];
	glm::vec4 indirect = dev_indirect[pixelIndex];

	// Remodulate with the albedo
	glm::vec3 albedo = dev_gBuffer_albedo[pixelIndex];
	// Use direct.w, but both have the same w value
	dev_outputColor[pixelIndex] = glm::vec4((glm::vec3(direct) + glm::vec3(indirect)) * albedo, direct.w);
}

void SVGFChannel::allocate(size_t numPixels)
{
	CUDA_CHECK(cudaMalloc(&dev_illuminationPrev, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_pingBuffer, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_pongBuffer, numPixels * sizeof(glm::vec4)));

	CUDA_CHECK(cudaMalloc(&dev_moments, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_momentsPrev, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_variancePing, numPixels * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dev_variancePong, numPixels * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dev_prefilteredVariance, numPixels * sizeof(float)));
}

void SVGFChannel::free()
{
	cudaFree(dev_illuminationPrev);
	cudaFree(dev_pingBuffer);
	cudaFree(dev_pongBuffer);
	cudaFree(dev_moments);
	cudaFree(dev_momentsPrev);
	cudaFree(dev_variancePing);
	cudaFree(dev_variancePong);
	cudaFree(dev_prefilteredVariance);

	*this = SVGFChannel{};
}

void SVGFChannel::swapBuffers()
{
	std::swap(dev_moments, dev_momentsPrev);
}

SVGFManager::SVGFManager()
{
}

SVGFManager::SVGFManager(int width, int height) :
	SVGFManager()
{
	resize(width, height);
}

SVGFManager::~SVGFManager()
{
	freeBuffers();
}

SVGFManager& SVGFManager::operator=(SVGFManager&& other) noexcept
{
	if (this != &other) {
		freeBuffers();

		m_width = other.m_width;
		m_height = other.m_height;

		dev_gBuffer_normalDepth = other.dev_gBuffer_normalDepth;
		dev_gBuffer_motionVectors = other.dev_gBuffer_motionVectors;
		dev_gBuffer_albedo = other.dev_gBuffer_albedo;
		dev_gBuffer_geometryId = other.dev_gBuffer_geometryId;
		dev_gBuffer_geometryIdPrev = other.dev_gBuffer_geometryIdPrev;
		dev_gBuffer_historyLength = other.dev_gBuffer_historyLength;
		dev_gBuffer_normalDepthPrev = other.dev_gBuffer_normalDepthPrev;
		dev_gBuffer_historyLengthPrev = other.dev_gBuffer_historyLengthPrev;
		directChannel = other.directChannel;
		indirectChannel = other.indirectChannel;
		dev_outputColor = other.dev_outputColor;

		other.m_width = 0;
		other.m_height = 0;
		other.dev_gBuffer_normalDepth = nullptr;
		other.dev_gBuffer_motionVectors = nullptr;
		other.dev_gBuffer_albedo = nullptr;
		other.dev_gBuffer_geometryId = nullptr;
		other.dev_gBuffer_geometryIdPrev = nullptr;
		other.dev_gBuffer_historyLength = nullptr;
		other.dev_gBuffer_normalDepthPrev = nullptr;
		other.dev_gBuffer_historyLengthPrev = nullptr;
		other.directChannel = SVGFChannel{};
		other.indirectChannel = SVGFChannel{};
		other.dev_outputColor = nullptr;

		m_prevViewProj = other.m_prevViewProj;
		m_svgfSettings = other.m_svgfSettings;
	}

	return *this;
}

SVGFManager::SVGFManager(SVGFManager&& other) noexcept
{
	*this = std::move(other);
}

void SVGFManager::swapBuffers()
{
	std::swap(dev_gBuffer_normalDepth, dev_gBuffer_normalDepthPrev);
	std::swap(dev_gBuffer_geometryId, dev_gBuffer_geometryIdPrev);
	std::swap(dev_gBuffer_historyLength, dev_gBuffer_historyLengthPrev);

	directChannel.swapBuffers();
	indirectChannel.swapBuffers();
}

void SVGFManager::resize(int width, int height)
{
	if (m_width == width && m_height == height) {
		return;
	}

	freeBuffers();

	m_width = width;
	m_height = height;

	// Reset previous viewproj since the camera was changed due to resizing
	m_prevViewProj.reset();

	if (m_width > 0 && m_height > 0) {
		// If width or height is 0, we don't allocate buffers
		allocateBuffers();
	}
}

void SVGFManager::captureGBuffer(
	const PathState* dev_pathStates,
	const IntersectionData* dev_intersectionData,
	int activePathCount,
	const Camera& camera,
	const std::unique_ptr<Scene>& scene)
{
	if (activePathCount <= 0) return;

	int blockSize = 256;
	int gridSize = divup(activePathCount, blockSize);

	glm::mat4 currentViewProj = camera.getViewProjectionMatrix(m_width, m_height);
	glm::mat4 prevViewProj = m_prevViewProj.has_value() ? m_prevViewProj.value() : currentViewProj;

	kernCaptureGBuffer << <gridSize, blockSize >> > (
		dev_pathStates,
		dev_intersectionData,
		dev_gBuffer_normalDepth,
		dev_gBuffer_motionVectors,
		dev_gBuffer_albedo,
		dev_gBuffer_geometryId,
		currentViewProj,
		prevViewProj,
		m_width,
		m_height,
		activePathCount,
		scene != nullptr ? scene->getDevScene() : DevScene{}
		);

	m_prevViewProj = currentViewProj;
}

void SVGFManager::evaluate()
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}
	demodulateAlbedo();
	executeTemporalAccumulation();
	executeVarianceEstimation();
	executeAtrousFilteringPipeline();
	combineChannels();
}

void SVGFManager::demodulateAlbedo()
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize = make2DGrid(m_width, m_height, blockSize);

	// First demodulate before accumulating
	kernDemodulateAlbedo << <gridSize, blockSize >> > (
		directChannel.dev_pingBuffer,
		indirectChannel.dev_pingBuffer,
		dev_gBuffer_albedo,
		m_width,
		m_height
		);
}

void SVGFManager::executeTemporalAccumulation()
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize = make2DGrid(m_width, m_height, blockSize);

	for (SVGFChannel* channel : { &directChannel, &indirectChannel }) {
		kernTemporalAccumulation << <gridSize, blockSize >> > (
			// Inputs
			dev_gBuffer_normalDepth, // Both channels share normal, motion, and history length buffers
			dev_gBuffer_motionVectors,
			dev_gBuffer_normalDepthPrev,
			dev_gBuffer_geometryId,
			dev_gBuffer_geometryIdPrev,
			channel->dev_pingBuffer,
			channel->dev_illuminationPrev,
			channel->dev_momentsPrev,
			// Outputs
			channel->dev_pongBuffer,
			channel->dev_moments,
			dev_gBuffer_historyLength,
			dev_gBuffer_historyLengthPrev,
			m_svgfSettings.colorAlpha,
			m_svgfSettings.momentsAlpha,
			m_width,
			m_height
			);

		// Swap ping and pong buffers so we can use this output as input for future operations
		std::swap(channel->dev_pingBuffer, channel->dev_pongBuffer);
	}
}

void SVGFManager::executeVarianceEstimation()
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize = make2DGrid(m_width, m_height, blockSize);

	for (SVGFChannel* channel : { &directChannel, &indirectChannel }) {
		kernEstimateVariance << <gridSize, blockSize >> > (
			channel->dev_pingBuffer,
			channel->dev_moments,
			dev_gBuffer_normalDepth, // Both channels share normal, motion, and history length buffers
			dev_gBuffer_historyLength,
			channel->dev_variancePing,
			m_width,
			m_height
			);
	}
}

void SVGFManager::executeAtrousFilteringPipeline() {
	if (m_width <= 0 || m_height <= 0) return;

	dim3 blockSize(16, 16);
	dim3 gridSize = make2DGrid(m_width, m_height, blockSize);

	size_t numPixels = static_cast<size_t>(m_width) * m_height;

	for (SVGFChannel* channel : { &directChannel, &indirectChannel }) {
		if (m_svgfSettings.atrousIterations <= 0) {
			// No wavelet iterations, so the temporally integrated color becomes next frame's color history
			CUDA_CHECK(cudaMemcpy(channel->dev_illuminationPrev, channel->dev_pingBuffer, numPixels * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
			continue;
		}

		for (int i = 0; i < m_svgfSettings.atrousIterations; ++i) {
			int stride = 1 << i;

			// Prefilter variance
			kernVariancePrefilter3x3 << <gridSize, blockSize >> > (
				channel->dev_variancePing,
				dev_gBuffer_normalDepth,
				channel->dev_prefilteredVariance,
				m_width, m_height
				);

			kernAtrousFilter << <gridSize, blockSize >> > (
				channel->dev_pingBuffer,
				dev_gBuffer_normalDepth,
				channel->dev_variancePing,
				channel->dev_prefilteredVariance,
				channel->dev_variancePong,
				channel->dev_pongBuffer,
				stride,
				m_svgfSettings.sigmaLuminance,
				m_svgfSettings.sigmaNormal,
				m_svgfSettings.sigmaDepth,
				m_width,
				m_height
				);

			if (i == 0) {
				// Save first iteration of wavelet to next frames color history
				CUDA_CHECK(cudaMemcpy(channel->dev_illuminationPrev, channel->dev_pongBuffer, numPixels * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
			}

			// Ping pong
			std::swap(channel->dev_pingBuffer, channel->dev_pongBuffer);
			std::swap(channel->dev_variancePing, channel->dev_variancePong);
		}
	}
}

void SVGFManager::combineChannels()
{
	if (m_width <= 0 || m_height <= 0) return;

	dim3 blockSize(16, 16);
	dim3 gridSize = make2DGrid(m_width, m_height, blockSize);

	kernCombineChannels << <gridSize, blockSize >> > (
		directChannel.dev_pingBuffer,
		indirectChannel.dev_pingBuffer,
		dev_gBuffer_albedo,
		dev_outputColor,
		m_width,
		m_height
		);
}

void SVGFManager::display(cudaSurfaceObject_t surface)
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize = make2DGrid(m_width, m_height, blockSize);

	switch (m_svgfSettings.view) {
	case SVGFView::Direct:
		kernDisplayChannel << <gridSize, blockSize >> > (directChannel.dev_pingBuffer, dev_gBuffer_albedo, surface, m_width, m_height);
		break;
	case SVGFView::Indirect:
		kernDisplayChannel << <gridSize, blockSize >> > (indirectChannel.dev_pingBuffer, dev_gBuffer_albedo, surface, m_width, m_height);
		break;
	case SVGFView::Albedo:
		kernDisplayAlbedo << <gridSize, blockSize >> > (dev_gBuffer_albedo, surface, m_width, m_height);
		break;
	case SVGFView::DirectVariance:
		kernDebugVariance << <gridSize, blockSize >> > (directChannel.dev_variancePing, surface, m_width, m_height);
		break;
	case SVGFView::IndirectVariance:
		kernDebugVariance << <gridSize, blockSize >> > (indirectChannel.dev_variancePing, surface, m_width, m_height);
		break;
	case SVGFView::HistoryLength:
		kernDisplayHistoryLength << <gridSize, blockSize >> > (dev_gBuffer_historyLength, surface, m_width, m_height);
		break;
	case SVGFView::Normals:
		kernDebugNormals << <gridSize, blockSize >> > (dev_gBuffer_normalDepth, surface, m_width, m_height);
		break;
	case SVGFView::MotionVectors:
		kernDebugMotionVectors << <gridSize, blockSize >> > (dev_gBuffer_motionVectors, surface, m_width, m_height);
		break;
	default:
		kernDebugSVGFIllumination << <gridSize, blockSize >> > (dev_outputColor, surface, m_width, m_height);
		break;
	}
}

void SVGFManager::drawSettingsImGui()
{
	const char* viewLabels[] = {
		"Final", "Direct", "Indirect", "Albedo",
		"Direct Variance", "Indirect Variance", "History Length",
		"Normals", "Motion Vectors"
	};
	int view = static_cast<int>(m_svgfSettings.view);
	if (ImGui::Combo("View", &view, viewLabels, IM_ARRAYSIZE(viewLabels))) {
		m_svgfSettings.view = static_cast<SVGFView>(view);
	}

	ImGui::SliderFloat("Color Alpha", &m_svgfSettings.colorAlpha, 0.01f, 1.0f);
	ImGui::SliderFloat("Moments Alpha", &m_svgfSettings.momentsAlpha, 0.01f, 1.0f);
	ImGui::SliderFloat("Sigma Luminance", &m_svgfSettings.sigmaLuminance, 0.1f, 16.0f);
	ImGui::SliderFloat("Sigma Normal", &m_svgfSettings.sigmaNormal, 1.0f, 256.0f);
	ImGui::SliderFloat("Sigma Depth", &m_svgfSettings.sigmaDepth, 0.01f, 4.0f);
	ImGui::SliderInt("A-trous Iterations", &m_svgfSettings.atrousIterations, 0, 8);
}

void SVGFManager::allocateBuffers()
{
	size_t numPixels = static_cast<size_t>(m_width) * m_height;

	CUDA_CHECK(cudaMalloc(&dev_gBuffer_normalDepth, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_normalDepthPrev, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_motionVectors, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_albedo, numPixels * sizeof(glm::vec3)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_geometryId, numPixels * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_geometryIdPrev, numPixels * sizeof(int)));

	// All bytes 0xFF makes every id -1, so the first frame never matches stale history
	CUDA_CHECK(cudaMemset(dev_gBuffer_geometryIdPrev, 0xFF, numPixels * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_historyLength, numPixels * sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_historyLengthPrev, numPixels * sizeof(unsigned int)));

	directChannel.allocate(numPixels);
	indirectChannel.allocate(numPixels);

	CUDA_CHECK(cudaMalloc(&dev_outputColor, numPixels * sizeof(glm::vec4)));

	// Initialize history tracker sizes to 0
	CUDA_CHECK(cudaMemset(dev_gBuffer_historyLength, 0, numPixels * sizeof(unsigned int)));
}

void SVGFManager::freeBuffers()
{
	if (dev_gBuffer_normalDepth) {
		cudaFree(dev_gBuffer_normalDepth);
	}

	if (dev_gBuffer_normalDepthPrev) {
		cudaFree(dev_gBuffer_normalDepthPrev);
	}

	if (dev_gBuffer_motionVectors) {
		cudaFree(dev_gBuffer_motionVectors);
	}

	if (dev_gBuffer_albedo) {
		cudaFree(dev_gBuffer_albedo);
	}

	cudaFree(dev_gBuffer_geometryId);
	cudaFree(dev_gBuffer_geometryIdPrev);

	if (dev_gBuffer_historyLength) {
		cudaFree(dev_gBuffer_historyLength);
	}

	if (dev_gBuffer_historyLengthPrev) {
		cudaFree(dev_gBuffer_historyLengthPrev);
	}

	if (dev_outputColor) {
		cudaFree(dev_outputColor);
	}

	directChannel.free();
	indirectChannel.free();

	dev_gBuffer_normalDepth = nullptr;
	dev_gBuffer_normalDepthPrev = nullptr;
	dev_gBuffer_motionVectors = nullptr;
	dev_gBuffer_albedo = nullptr;
	dev_gBuffer_geometryId = nullptr;
	dev_gBuffer_geometryIdPrev = nullptr;
	dev_gBuffer_historyLength = nullptr;
	dev_gBuffer_historyLengthPrev = nullptr;
	dev_outputColor = nullptr;
}
