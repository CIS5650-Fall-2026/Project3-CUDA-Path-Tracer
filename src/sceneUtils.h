#pragma once

// Turns parsed scene data into the flat, GPU-ready tables consumed
// by Vulkan HW RT

#include "gpu/shared.h"
#include "sceneStructs.h"

#include <cstdint>
#include <vector>

class Scene;

namespace sceneutil
{
    // We'll use this to help build the BLAS,
    // and each MeshPrimitive will have one
    // associated MeshPrimitiveRange struct
    struct MeshPrimitiveRange
    {
        // This offsets into the FlatScene vertex buffer
        uint32_t vertexOffset;
        uint32_t vertexCount;

        // This offsets into the FlatScene index buffer
        uint32_t indexOffset;
        uint32_t indexCount;
    };

    struct FlatScene
    {
        std::vector<gpu::Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<gpu::GeometryInfo> geometries;
        std::vector<gpu::GpuMaterial> materials;
        std::vector<gpu::EmissiveTri> lights;
        float totalLightArea = 0;

        // Indexed like [meshIndex][primitiveIndex]
        std::vector<std::vector<MeshPrimitiveRange>> meshRanges;

        // Indexed like [instanceIndex]
        // These values become each instance's InstanceCustomIndexKHR value
        std::vector<uint32_t> instanceFirstGeometry;
        std::vector<MeshInstance> instances;
    };

    // Converts Scene --> FlatScene
    FlatScene flattenScene(const Scene& scene);

} // namespace sceneutil