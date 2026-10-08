#include "microfacet.h"

__device__ float MicrofacetScattering::reflectionProbability(float dotIH, float etaI, float etaT) const {
	if (!bAllowTransmission) {
		// No transmission allowed, always reflect
		return 1.0f;
	}

	float eta = etaI / etaT;
	float sin2ThetaT = eta * eta * glm::max(0.0f, 1.0f - dotIH * dotIH);
	if (sin2ThetaT >= 1.0f) {
		// Total internal reflection, every sample reflects
		return 1.0f;
	}

	float F = Microfacet::F_SchlickDielectric(dotIH, etaI, etaT);
	return glm::clamp(F, 0.10f, 0.90f);
}

__device__ glm::vec3 MicrofacetScattering::evaluate(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside) const {
	float cosThetaI = glm::max(0.0f, glm::dot(n, wi));
	float cosThetaO = glm::dot(n, wo);

	if (cosThetaI <= 0.0f) {
		return glm::vec3(0.0f);
	}

	bool bIsReflection = cosThetaO > 0.0f;

	float absCosThetaO = glm::abs(cosThetaO);

	if (absCosThetaO <= 0.0f) {
		return glm::vec3(0.0f);
	}

	float etaI = bInside ? ior : 1.0f;
	float etaT = bInside ? 1.0f : ior;

	if (bIsReflection) {
		glm::vec3 H = glm::normalize(wi + wo);
		float dotNH = glm::max(0.0f, glm::dot(n, H));
		float dotHO = glm::max(0.0f, glm::dot(H, wo));

		float D = (model == DistributionModel::BlinnPhong)
			? Microfacet::D_BlinnPhong(dotNH, specularExponent)
			: Microfacet::D_GGX(dotNH, alpha);

		float G = (model == DistributionModel::BlinnPhong)
			? Microfacet::G_VCavity(dotNH, cosThetaI, absCosThetaO, dotHO)
			: Microfacet::G_SmithGGX(cosThetaI, absCosThetaO, alpha);

		glm::vec3 F = Microfacet::F_Schlick(dotHO, f0);

		return (D * G * F) / (4.0f * cosThetaI * absCosThetaO);
	}
	else {
		glm::vec3 H = glm::normalize(wi + wo * (etaT / etaI));
		if (glm::dot(n, H) < 0.0f) {
			H = -H;
		}

		float dotNH = glm::max(0.0f, glm::dot(n, H));
		float dotIH = glm::max(0.0f, glm::dot(wi, H));
		float dotOH = glm::dot(wo, H);
		float absDotOH = glm::abs(dotOH);

		float D = (model == DistributionModel::BlinnPhong)
			? Microfacet::D_BlinnPhong(dotNH, specularExponent)
			: Microfacet::D_GGX(dotNH, alpha);

		float G = (model == DistributionModel::BlinnPhong)
			? Microfacet::G_VCavity(dotNH, cosThetaI, absCosThetaO, absDotOH)
			: Microfacet::G_SmithGGX(cosThetaI, absCosThetaO, alpha);

		float F = Microfacet::F_SchlickDielectric(dotIH, etaI, etaT);

		float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
		if (glm::abs(sqrtDenom) < 1e-6f) {
			return glm::vec3(0.0f);
		}

		float transmissionValue = ((absDotOH * dotIH) * D * G * (1.0f - F)) /
			(cosThetaI * absCosThetaO * sqrtDenom * sqrtDenom);

		return transmissionTint * transmissionValue;
	}
}

__device__ float MicrofacetScattering::pdf(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside) const {
	float cosThetaI = glm::dot(n, wi);
	float cosThetaO = glm::dot(n, wo);
	if (cosThetaI <= 0.0f) {
		return 0.0f;
	}

	bool bIsReflection = cosThetaO > 0.0f;
	float absCosThetaO = glm::abs(cosThetaO);
	if (absCosThetaO <= 0.0f) {
		return 0.0f;
	}

	float etaI = bInside ? ior : 1.0f;
	float etaT = bInside ? 1.0f : ior;

	glm::vec3 H;
	if (bIsReflection) {
		H = glm::normalize(wi + wo);
	}
	else {
		H = glm::normalize(wi + wo * (etaT / etaI));
		if (glm::dot(n, H) < 0.0f) H = -H;
	}

	float dotNH = glm::max(0.0f, glm::dot(n, H));
	float dotIH = glm::max(0.0f, glm::dot(wi, H));

	float pdfHalf = (model == DistributionModel::BlinnPhong)
		? Microfacet::PDF_BlinnPhong(dotNH, specularExponent)
		: Microfacet::PDF_GGX(dotNH, alpha);

	float reflectionProb = reflectionProbability(dotIH, etaI, etaT);

	if (bIsReflection) {
		if (dotIH <= 0.0f) {
			return 0.0f;
		}
		return (pdfHalf / (4.0f * dotIH)) * reflectionProb;
	}
	else {
		float dotOH = glm::dot(wo, H);
		float absDotOH = glm::abs(dotOH);
		float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
		if (glm::abs(sqrtDenom) < 1e-6f) {
			return 0.0f;
		}

		// Matching pdf with sample() for transmission
		float transmissionProb = 1.0f - reflectionProb;
		return (pdfHalf * (etaT * etaT / (etaI * etaI)) * absDotOH / (sqrtDenom * sqrtDenom)) * transmissionProb;
	}
}

__device__ void MicrofacetScattering::sample(
	const glm::vec3& n, const glm::vec3& wi,
	const glm::vec3& random, bool bInside,
	glm::vec3& out_wo, float& out_pdf, bool& out_isTransmission) const
{
	float cosThetaI = glm::dot(n, wi);
	if (cosThetaI <= 0.0f) {
		// Incoming vector is behind the material, fail
		out_pdf = 0.0f;
		return;
	}

	// Sample the half vector in local space
	glm::vec3 H_local = (model == DistributionModel::GGX)
		? Microfacet::Sample_GGX(glm::vec2(random), alpha)
		: Microfacet::Sample_BlinnPhong(glm::vec2(random), specularExponent);

	// Transform half vector to world space
	glm::vec3 tangent, bitangent;
	MathHelpers::createCoordinateSystem(n, tangent, bitangent);
	glm::vec3 H = glm::normalize(tangent * H_local.x + bitangent * H_local.y + n * H_local.z);

	float dotIH = glm::max(0.0f, glm::dot(wi, H));
	float etaI = bInside ? ior : 1.0f;
	float etaT = bInside ? 1.0f : ior;

	float reflectionProb = reflectionProbability(dotIH, etaI, etaT);

	float dotNH = glm::max(0.0f, glm::dot(n, H));
	float pdfHalf = (model == DistributionModel::GGX)
		? Microfacet::PDF_GGX(dotNH, alpha)
		: Microfacet::PDF_BlinnPhong(dotNH, specularExponent);

	if (random.z < reflectionProb) {
		// Reflect
		out_isTransmission = false;
		out_wo = glm::reflect(-wi, H);

		float cosThetaO = glm::dot(n, out_wo);
		if (cosThetaO <= 0.0f || dotIH <= 0.0f) {
			out_pdf = 0.0f;
			return;
		}

		glm::vec3 f = evaluate(n, wi, out_wo, bInside);
		out_pdf = (pdfHalf / (4.0f * dotIH)) * reflectionProb;
	}
	else {
		// Refract
		float eta = etaI / etaT;
		glm::vec3 refract_vector = glm::refract(-wi, H, eta);

		// Total internal reflection
		if (glm::dot(refract_vector, refract_vector) < 0.0001f) {
			out_isTransmission = false;
			out_wo = glm::reflect(-wi, H);

			float cosThetaO = glm::dot(n, out_wo);
			if (cosThetaO <= 0.0f || dotIH <= 0.0f) {
				out_pdf = 0.0f;
				return;
			}

			// Reflection probability is 1.0, so no need to multiply the pdf by reflectionProb
			out_pdf = pdfHalf / (4.0f * dotIH);
			return;
		}

		out_isTransmission = true;
		out_wo = glm::normalize(refract_vector);

		float cosThetaO = glm::dot(n, out_wo);
		float absCosThetaO = glm::abs(cosThetaO);
		float dotOH = glm::dot(out_wo, H);
		float absDotOH = glm::abs(dotOH);

		float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
		if (glm::abs(sqrtDenom) < 1e-6f) {
			out_pdf = 0.0f;
			return;
		}

		float transmissionProb = 1.0f - reflectionProb;

		out_pdf = (pdfHalf * (etaT * etaT / (etaI * etaI)) * absDotOH / (sqrtDenom * sqrtDenom)) * transmissionProb;
	}
}