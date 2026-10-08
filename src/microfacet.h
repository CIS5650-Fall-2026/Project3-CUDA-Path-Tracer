#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "mathHelpers.h"

enum class DistributionModel {
	BlinnPhong = 0,
	GGX = 1
};

namespace Microfacet {
	// Normal Distribution functions
	__device__ inline float D_BlinnPhong(float dotNH, float specularExponent) {
		if (dotNH <= 0.0f) {
			return 0.0f;
		}

		return ((specularExponent + 2.0f) / (2.0f * glm::pi<float>())) * powf(dotNH, specularExponent);
	}

	__device__ inline float D_GGX(float dotNH, float alpha) {
		if (dotNH <= 0.0f) {
			return 0.0f;
		}

		float alpha2 = alpha * alpha;
		float denom = dotNH * dotNH * (alpha2 - 1.0f) + 1.0f;
		return alpha2 / (glm::pi<float>() * denom * denom);
	}

	// PDF functions
	__device__ inline float PDF_BlinnPhong(float dotNH, float specularExponent) {
		return D_BlinnPhong(dotNH, specularExponent) * dotNH;
	}

	__device__ inline float PDF_GGX(float dotNH, float alpha) {
		return D_GGX(dotNH, alpha) * dotNH;
	}

	// Geometry functions
	__device__ inline float G_VCavity(float dotNH, float dotNV, float cosThetaO, float dotLH) {
		if(dotLH <= 0.0f) {
			return 0.0f;
		}
		return glm::min(1.0f, glm::min(
			(2.0f * dotNH * dotNV) / dotLH, 
			(2.0f * dotNH * cosThetaO) / dotLH)
		);
	}

	__device__ inline float G1_SmithGGX(float dotNX, float alpha) {
		if (dotNX <= 0.0f) {
			return 0.0f;
		}

		float alpha2 = alpha * alpha;
		float dotNX2 = dotNX * dotNX;
		return 2.0f * dotNX / (dotNX + glm::sqrt(alpha2 + (1.0f - alpha2) * dotNX2));
	}

	__device__ inline float G_SmithGGX(float dotNV, float cosThetaO, float alpha) {
		return G1_SmithGGX(dotNV, alpha) * G1_SmithGGX(cosThetaO, alpha);
	}

	// Fresnel
	__device__ inline glm::vec3 F_Schlick(float dotLH, const glm::vec3& f0) {
		return f0 + (glm::vec3(1.0f) - f0) * glm::pow(1.0f - glm::max(0.0f, dotLH), 5.0f);
	}

	// Schlick Fresnel for traversing from etaI into etaT
	__device__ inline float F_SchlickDielectric(float cosTheta, float etaI, float etaT) {
		float r0 = (etaI - etaT) / (etaI + etaT);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * glm::pow(1.0f - glm::max(0.0f, cosTheta), 5.0f);
	}

	// Samplers
	__device__ inline glm::vec3 Sample_BlinnPhong(const glm::vec2& random, float specularExponent) {
		float cosTheta = glm::pow(random.x, 1.0f / (specularExponent + 1.0f));
		float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
		float phi = 2.0f * glm::pi<float>() * random.y;

		return glm::vec3(
			sinTheta * glm::cos(phi),
			sinTheta * glm::sin(phi),
			cosTheta
		);
	}

	__device__ inline glm::vec3 Sample_GGX(const glm::vec2& random, float alpha) {
		float alpha2 = alpha * alpha;
		float cosTheta = glm::sqrt(glm::max(0.0f, (1.0f - random.x) / (random.x * (alpha2 - 1.0f) + 1.0f)));
		float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
		float phi = 2.0f * glm::pi<float>() * random.y;
		return glm::vec3(
			sinTheta * glm::cos(phi), 
			sinTheta * glm::sin(phi), 
			cosTheta
		);
	}
};

struct MicrofacetScattering {
	DistributionModel model;

	union
	{
		float specularExponent; // For Blinn-Phong
		float alpha; // For GGX
	};

	glm::vec3 f0; // Base reflectivity at normal incidence
	glm::vec3 transmissionTint = glm::vec3(1.0f); // Tints refracted light
	float ior;

	bool bAllowTransmission = false;

	// Probability of reflecting vs. refracting off of a sampled microfacet
	__device__ float reflectionProbability(float dotIH, float etaI, float etaT) const;

	__device__ glm::vec3 evaluate(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside) const;

	__device__ float pdf(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside) const;

	// Sampling function for phong and pbr materials
	__device__ void sample(
		const glm::vec3& n, const glm::vec3& wi,
		const glm::vec3& random, bool bInside,
		glm::vec3& out_wo, float& out_pdf, bool& out_isTransmission) const;
};