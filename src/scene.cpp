#include "scene.h"

#include "bvh.h"
#include "material.h"
#include "pathTraceCommon.h"
#include "textureLoader.h"
#include "cudaHelpers.h"

// External Includes
#include <cuda_runtime.h>

Scene::Scene()
{
}

Scene::Scene(const std::vector<Geom>& geometry,
    int lightCount,
    const std::vector<Triangle>& triangles, 
    const std::vector<BLASNode>& blasNodes, 
    const std::vector<TLASNode>& tlasNodes, 
    const std::vector<Material>& materials, 
    const std::vector<cudaTextureObject_t>& textures, 
    const std::vector<cudaArray_t>& textureArrays) :
	m_geometryCount(geometry.size()),
	m_lightCount(lightCount),
	m_triangleCount(triangles.size()),
	m_blasNodeCount(blasNodes.size()),
	m_tlasNodeCount(tlasNodes.size()),
	m_materialCount(materials.size()),
	m_textures(textures),
	m_textureArrays(textureArrays)
{
    if (!geometry.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&dev_geometry, geometry.size() * sizeof(Geom)));
        CUDA_CHECK(cudaMemcpy(dev_geometry, geometry.data(), geometry.size() * sizeof(Geom), cudaMemcpyHostToDevice));
    }

    // Allocate materials
    if (!materials.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&dev_materials, materials.size() * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(dev_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    }

    // Allocate triangles
    if (!triangles.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&dev_triangles, triangles.size() * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(dev_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    }

    // Allocate BLAS nodes
    if (!blasNodes.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&dev_blasNodes, blasNodes.size() * sizeof(BLASNode)));
        CUDA_CHECK(cudaMemcpy(dev_blasNodes, blasNodes.data(), blasNodes.size() * sizeof(BLASNode), cudaMemcpyHostToDevice));
    }

    // Allocate TLAS nodes
    if (!tlasNodes.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&dev_tlasNodes, tlasNodes.size() * sizeof(TLASNode)));
        CUDA_CHECK(cudaMemcpy(dev_tlasNodes, tlasNodes.data(), tlasNodes.size() * sizeof(TLASNode), cudaMemcpyHostToDevice));
    }
}

Scene::~Scene()
{
    if (dev_geometry) {
        cudaFree(dev_geometry);
        dev_geometry = nullptr;
    }

    if (dev_triangles) {
        cudaFree(dev_triangles);
        dev_triangles = nullptr;
    }

    if (dev_blasNodes) {
        cudaFree(dev_blasNodes);
        dev_blasNodes = nullptr;
    }

    if (dev_tlasNodes) {
        cudaFree(dev_tlasNodes);
        dev_tlasNodes = nullptr;
    }

    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }

    for(auto texture : m_textures) {
        if(texture) {
            cudaDestroyTextureObject(texture);
        }
	}
    m_textures.clear();

    for(auto textureArray : m_textureArrays) {
        if (textureArray) {
            cudaFreeArray(textureArray);
        }
    }
	m_textureArrays.clear();
}

Scene& Scene::operator=(Scene&& other) noexcept
{
    if (this != &other) {
		freeDeviceMemory();

        // Move from other scene
        dev_geometry = other.dev_geometry;
        m_geometryCount = other.m_geometryCount;
        dev_materials = other.dev_materials;
        m_materialCount = other.m_materialCount;
        dev_triangles = other.dev_triangles;
        m_triangleCount = other.m_triangleCount;
		dev_blasNodes = other.dev_blasNodes;
		m_blasNodeCount = other.m_blasNodeCount;
        dev_tlasNodes = other.dev_tlasNodes;
		m_tlasNodeCount = other.m_tlasNodeCount;
		m_textures = std::move(other.m_textures);
		m_textureArrays = std::move(other.m_textureArrays);
		
        // Clear other scene
        other.dev_geometry = nullptr;
        other.m_geometryCount = 0;
        other.dev_materials = nullptr;
        other.m_materialCount = 0;
		other.dev_triangles = nullptr;
		other.m_triangleCount = 0;
        other.dev_blasNodes = nullptr;
        other.m_blasNodeCount = 0;
		other.dev_tlasNodes = nullptr;
		other.m_tlasNodeCount = 0;
    }

    return *this;
}

Scene::Scene(Scene&& other) noexcept :
    dev_geometry(other.dev_geometry),
    m_geometryCount(other.m_geometryCount),
    dev_materials(other.dev_materials),
    m_materialCount(other.m_materialCount),
    dev_triangles(other.dev_triangles),
    m_triangleCount(other.m_triangleCount),
    dev_blasNodes(other.dev_blasNodes),
    m_blasNodeCount(other.m_blasNodeCount),
	dev_tlasNodes(other.dev_tlasNodes),
	m_tlasNodeCount(other.m_tlasNodeCount),
    m_textures(std::move(other.m_textures)),
	m_textureArrays(std::move(other.m_textureArrays))
{
    other.dev_geometry = nullptr;
    other.m_geometryCount = 0;
    other.dev_materials = nullptr;
    other.m_materialCount = 0;
    other.dev_triangles = nullptr;
    other.m_triangleCount = 0;
    other.dev_blasNodes = nullptr;
    other.m_blasNodeCount = 0;
    other.dev_tlasNodes = nullptr;
    other.m_tlasNodeCount = 0;
}

void Scene::freeDeviceMemory()
{
    if (dev_geometry) {
        cudaFree(dev_geometry);
        dev_geometry = nullptr;
    }

    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }

    if (dev_triangles) {
        cudaFree(dev_triangles);
        dev_triangles = nullptr;
    }

    if (dev_blasNodes) {
        cudaFree(dev_blasNodes);
        dev_blasNodes = nullptr;
    }

    for (auto texture : m_textures) {
        if (texture) {
            cudaDestroyTextureObject(texture);
        }
    }
    m_textures.clear();

    for (auto textureArray : m_textureArrays) {
        if (textureArray) {
            cudaFreeArray(textureArray);
        }
    }
    m_textureArrays.clear();
}
