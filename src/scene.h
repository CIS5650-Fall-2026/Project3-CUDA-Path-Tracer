#pragma once

#include "pathTraceCommon.h"
#include "devScene.h"

#include <vector>

struct Material;
struct BLASNode;
struct TLASNode;
struct Geom;
struct Triangle;

class Scene {
public:
	Scene();
	Scene(const std::vector<Geom>& geometry,
		int lightCount,
		const std::vector<Triangle>& triangles,
		const std::vector<BLASNode>& blasNodes,
		const std::vector<TLASNode>& tlasNodes,
		const std::vector<Material>& materials,
		const std::vector<cudaTextureObject_t>& textures = {},
		const std::vector<cudaArray_t>& textureArrays = {});
	~Scene();

	Scene& operator=(const Scene&) = delete;
	Scene(const Scene&) = delete;

	Scene& operator=(Scene&&) noexcept;
	Scene(Scene&&) noexcept;

	DevScene getDevScene() {
		return DevScene{
			dev_geometry,
			m_geometryCount,
			dev_triangles,
			m_triangleCount,
			dev_blasNodes,
			m_blasNodeCount,
			dev_tlasNodes,
			m_tlasNodeCount,
			dev_materials,
			m_materialCount,
			m_lightCount
		};
	}

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;

	// First m_lightCount pieces of geometry are lights
	size_t m_lightCount = 0;

	Triangle* dev_triangles = nullptr;
	size_t m_triangleCount = 0;

	BLASNode* dev_blasNodes = nullptr;
	size_t m_blasNodeCount = 0;

	TLASNode* dev_tlasNodes = nullptr;
	size_t m_tlasNodeCount = 0;

	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;

	// CPU tracking of textures for memory management
	std::vector<cudaTextureObject_t> m_textures;
	std::vector<cudaArray_t> m_textureArrays;

private:
	void freeDeviceMemory();
};