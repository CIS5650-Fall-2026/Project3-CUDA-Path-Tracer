#include "material.h"

#include "samplers.h"
#include "devScene.h"
#include "intersection.h"

__device__ void Material::initializeFromIntersection(const IntersectionData& intersectionData)
{
	float u = intersectionData.uv.x;
	float v = intersectionData.uv.y;

	if(albedoTexture != 0) {
		float4 texColor = tex2D<float4>(albedoTexture, u, v);

		// Linearize albedo color
		albedo.x = powf(texColor.x, 2.2f);
		albedo.y = powf(texColor.y, 2.2f);
		albedo.z = powf(texColor.z, 2.2f);
	}

	if(type == MaterialType::PbrMetallicRoughness && pbr.metallicRoughnessTexture != 0) {
		float4 texColor = tex2D<float4>(pbr.metallicRoughnessTexture, u, v);

		// Format is red is unused, green is roughness, and blue is metallic
		pbr.roughness *= texColor.y;
		pbr.metallic *= texColor.z;
	}
}

__device__ glm::vec3 Material::evaluate(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside)
{
	if (isDelta()) {
		return glm::vec3(0.0f);
	}

	float cosThetaI = glm::dot(n, wi);
	float cosThetaO = glm::dot(n, wo);
	if (cosThetaI <= 0.0f) {
		return glm::vec3(0.0f);
	}

	bool bIsReflection = cosThetaO > 0.0f;
	float absCosThetaO = glm::abs(cosThetaO);

	if (type == MaterialType::OpaqueDiffuse) {
		if(!bIsReflection) {
			return glm::vec3(0.0f);
		}
		return albedo / glm::pi<float>();
	}

	MicrofacetScattering scattering = getSpecularScattering();
	glm::vec3 specularComponent = scattering.evaluate(n, wi, wo, bInside);

	if (type == MaterialType::PbrMetallicRoughness) {
		if (bIsReflection) {
			glm::vec3 H = glm::normalize(wi + wo);
			float dotIH = glm::max(0.0f, glm::dot(wi, H));
			float etaI = bInside ? ior : 1.0f;
			float etaT = bInside ? 1.0f : ior;
			float F = Microfacet::F_SchlickDielectric(dotIH, etaI, etaT);

			glm::vec3 diffuseComponent = (1.0f - F) * (albedo / glm::pi<float>()) * (1.0f - pbr.metallic) * (1.0f - pbr.transmission);
			return diffuseComponent + specularComponent;
		}
		else {
			return specularComponent * pbr.transmission;
		}
	}
	else {
		if(!bIsReflection) {
			return glm::vec3(0.0f);
		}

		glm::vec3 diffuseComponent = (albedo / glm::pi<float>());
      	return diffuseComponent + specularComponent;
	}
}

__device__ float Material::pdf(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside)
{
	if(isDelta()) {
		return 0.0f;
	}

	float cosThetaI = glm::dot(n, wi);
	float cosThetaO = glm::dot(n, wo);
	if(cosThetaI <= 0.0f) {
		return 0.0f;
	}

	bool bIsReflection = cosThetaO > 0.0f;
	float absCosThetaO = glm::abs(cosThetaO);

	if (type == MaterialType::OpaqueDiffuse) {
		if(!bIsReflection) {
			return 0.0f;
		}
		return absCosThetaO / glm::pi<float>();
	}

	// Incorporate diffuse probability since we randomly pick between diffuse and microfacet sampling
	float diffuseProbability = getDiffuseSampleProbability();
	float diffusePdf = bIsReflection ? absCosThetaO / glm::pi<float>() : 0.0f;

	MicrofacetScattering scattering = getSpecularScattering();
	float microfacetPdf = scattering.pdf(n, wi, wo, bInside);

	return diffuseProbability * diffusePdf + (1.0f - diffuseProbability) * microfacetPdf;
}

__device__ void Material::sample(
	const glm::vec3& n,
	const glm::vec3& wi,
	const glm::vec3& random,
	bool bInside,
	glm::vec3& wo,
	glm::vec3& outThroughput,
	float& outPdf,
	bool& bIsTransmission)
{
	float cosThetaI = glm::dot(n, wi);
	if (cosThetaI <= 0.0f) {
		// Incoming vector is opposite the normal, fail to sample
		outPdf = 0.0f;
		outThroughput = glm::vec3(0.0f);
		return;
	}

	if (isDelta()) {
		if (type == MaterialType::PerfectSpecular && !blinnPhong.bRefractive) {
			// Mirror with no transmission, sample direction is always pure reflection
			wo = glm::reflect(-wi, n);
			outPdf = 1.0f;
			outThroughput = albedo;
			bIsTransmission = false;
			return;
		}

		// Glass, could either reflect or refract
		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;
		// Get reflection probability using Schlick's approximation
		float F = Microfacet::F_SchlickDielectric(cosThetaI, etaI, etaT);

		float eta = etaI / etaT;
		glm::vec3 refracted = glm::refract(-wi, n, eta);

		// Check for total internal reflection. glm::refract will return a zero vector if this happens
		bool bTIR = (glm::dot(refracted, refracted) <= 0.0f);

		// Only perfect specular or blinn phong can be delta, so just take albedo as throughput
		outThroughput = albedo;

		if(random.z < F || bTIR) {
			wo = glm::reflect(-wi, n);
			outPdf = bTIR ? 1.0f : F; // TIR forces reflection, so the pdf in that case is 1.0
			bIsTransmission = false;
		}
		else {
			wo = glm::normalize(refracted);
			outPdf = 1.0f - F;
			bIsTransmission = true;
		}

		return;
	}

	if (type == MaterialType::OpaqueDiffuse) {
		// Simple diffuse material, use cosine weighted hemisphere sampling
		wo = Samplers::sampleCosineWeightedHemisphere(n, glm::vec2(random), outPdf);
		outThroughput = albedo;
		bIsTransmission = false;
		return;
	}

	// Pick between diffuse and microfacet sampling for faster convergence
	float diffuseProbability = getDiffuseSampleProbability();

	if (random.z < diffuseProbability) {
		// Use cosine weighted hemisphere sampling for diffuse reflection
		float diffusePdf = 0.0f;
		wo = Samplers::sampleCosineWeightedHemisphere(n, glm::vec2(random), diffusePdf);
	}
	else {
		// Use microfacet sampling
		glm::vec3 remappedRandom = glm::vec3(random.x, random.y, (random.z - diffuseProbability) / (1.0f - diffuseProbability));

		MicrofacetScattering scattering = getSpecularScattering();
		// Dummy value since we will recalculate the pdf after
		float lobePdf = 0.0f;
		scattering.sample(n, wi, remappedRandom, bInside, wo, lobePdf, bIsTransmission);

		if (lobePdf <= 0.0f) {
			outPdf = 0.0f;
			outThroughput = glm::vec3(0.0f);
			return;
		}
	}

	float cosThetaO = glm::dot(n, wo);
	bIsTransmission = cosThetaO < 0.0f;

	outPdf = pdf(n, wi, wo, bInside);
	if (outPdf <= 0.0f) {
		outThroughput = glm::vec3(0.0f);
		return;
	}

	outThroughput = evaluate(n, wi, wo, bInside) * glm::abs(cosThetaO) / outPdf;
}
