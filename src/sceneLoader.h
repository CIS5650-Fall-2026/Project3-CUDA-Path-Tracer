#pragma once

#include <glm/glm.hpp>
#include "material.h"

#include <memory>
#include <string>

class Scene;

enum class UpAxis {
	YUp,
	ZUp,
	XUp
};

struct FallbackMaterialSettings {
	enum class Type : int {
		Diffuse = 0,
		Mirror = 1,
		Glass = 2,
		Glossy = 3, // Blinn-Phong
		Pbr = 4 // GGX metallic-roughness
	};

	Type type = Type::Diffuse;
	glm::vec3 albedo = glm::vec3(0.8f);
	float roughness = 0.5f; // Used with Glossy and PBR
	float metallic = 0.0f; // Used with PBR
	float ior = 1.5f; // Used with Glass

	Material toMaterial() const;
};

struct SceneLoadOptions {
	UpAxis upAxis = UpAxis::YUp;

	// Some models are really small so this lets us scale them up when importing
	float importScale = 1.0f;

	FallbackMaterialSettings fallbackMaterial;

	// Replace every non-emissive material with the fallback
	bool bOverrideMaterials = false;
};

class SceneLoader {
public:
	static std::unique_ptr<Scene> loadFromFile(const std::string& filepath, const SceneLoadOptions& options = SceneLoadOptions{});

private:
	static std::unique_ptr<Scene> loadFromJson(const std::string& filepath, const SceneLoadOptions& options);
	static std::unique_ptr<Scene> loadFromGltf(const std::string& filepath, const SceneLoadOptions& options);
	static std::unique_ptr<Scene> loadFromObj(const std::string& filepath, const SceneLoadOptions& options);
};