#pragma once

#include <glm/glm.hpp>

#include "pathTraceCommon.h"
#include "microfacet.h"

struct DevScene;
struct IntersectionData;

enum class MaterialType : int {
    OpaqueDiffuse = 0, // Simple diffuse material
	PerfectSpecular = 1, // Mirror or glass material. Uses blinnPhong.bRefractive to distinguish
	BlinnPhong = 2, // Glossy specular material using Blinn-Phong model
	PbrMetallicRoughness = 3, // Physically based rendering material using metallic-roughness workflow
};

struct Material
{
	MaterialType type = MaterialType::OpaqueDiffuse;
	glm::vec3 albedo = glm::vec3(1.0, 0.0, 0.203);
	cudaTextureObject_t albedoTexture = 0;
    float emittance = 0.0f;
    float ior = 1.0f;

    union {
        // Obj / json format material type
        struct {
			glm::vec3 specularColor = glm::vec3(1.0, 0.0, 0.203);
            float exponent = 1.0f;
			bool bRefractive = false;
        } blinnPhong;

		// glTF format material type
        struct {
            float metallic = 0.0f;
            float roughness = 1.0f;
            float transmission = 0.0f;
			cudaTextureObject_t metallicRoughnessTexture = 0;
        } pbr;
	};

    __device__ bool isDelta() const {
        if (type == MaterialType::PerfectSpecular) {
            return true;
        }
        if(type == MaterialType::BlinnPhong && blinnPhong.exponent > 1000.0f) {
            return true;
		}
        // PBR materials have a minimum roughness applied so they are not a delta distribution
        return false;
	}

    // For non-diffuse materials, weight some specular color for the albedo as well
    __device__ glm::vec3 getDemodulationAlbedo() const {
        if (type == MaterialType::OpaqueDiffuse) {
            return albedo;
        }

        float diffuseProbability = getDiffuseSampleProbability();
        glm::vec3 specularAlbedo = getSpecularScattering().f0;
        return diffuseProbability * albedo + (1.0f - diffuseProbability) * specularAlbedo;
    }

    __device__ float getDiffuseSampleProbability() const {
        if (type == MaterialType::PbrMetallicRoughness) {
			// 50/50 split between diffuse and specular, but if metal or transmissive then no diffuse sampling
            return 0.5f * (1.0f - pbr.metallic) * (1.0f - pbr.transmission);
        }

        if (type == MaterialType::BlinnPhong) {
            // Weight by specular vs diffuse intensity
            const glm::vec3 luminanceWeights = glm::vec3(0.2126f, 0.7152f, 0.0722f);
            float diffuseWeight = glm::dot(albedo, luminanceWeights);
            float specularWeight = glm::dot(blinnPhong.specularColor, luminanceWeights);
            float totalWeight = diffuseWeight + specularWeight;
            if (totalWeight <= 0.0f) {
                return 0.5f;
            }
            return glm::clamp(diffuseWeight / totalWeight, 0.1f, 0.9f);
        }

        return 1.0f;
    }

    __device__ void initializeFromIntersection(const IntersectionData& intersectionData);

	__device__ glm::vec3 evaluate(
        const glm::vec3& n,
        const glm::vec3& wi,
        const glm::vec3& wo,
        bool bInside);

    __device__ float pdf(
        const glm::vec3& n,
        const glm::vec3& wi,
        const glm::vec3& wo,
        bool bInside);

    __device__ void sample(
        const glm::vec3& n, 
        const glm::vec3& wi, 
        const glm::vec3& random, 
        bool bInside, 
        glm::vec3& wo, 
        glm::vec3& outThroughput, 
        float& outPdf, 
		bool& bIsTransmission);

    __device__ MicrofacetScattering getSpecularScattering() const {
		MicrofacetScattering scattering{};
        scattering.ior = ior;

        if(type == MaterialType::BlinnPhong) {
            scattering.model = DistributionModel::BlinnPhong;
            scattering.specularExponent = blinnPhong.exponent;
            scattering.f0 = blinnPhong.specularColor;
        }
        else if(type == MaterialType::PbrMetallicRoughness) {
            scattering.model = DistributionModel::GGX;
			// Clamp roughness to a minimum value so we don't get a delta distribution
            float roughness = glm::max(pbr.roughness, 0.05f);
            scattering.alpha = roughness * roughness;

            // Dielectrics reflect a color set by their IOR, metals reflect their base color
            float r0 = (ior - 1.0f) / (ior + 1.0f);
            r0 = r0 * r0;
            scattering.f0 = glm::mix(glm::vec3(r0), albedo, pbr.metallic);

            // Refracted light is tinted by the base color
            scattering.transmissionTint = albedo;
            scattering.bAllowTransmission = pbr.transmission > 0.0f;
		}

        return scattering;
    }
};