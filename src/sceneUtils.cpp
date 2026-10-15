#include "sceneUtils.h"

#include "scene.h"

namespace sceneutil
{
    // Helper to add a mesh to the flatenned scene
    void addMesh(const MeshData& mesh, FlatScene& flatScene)
    { 
        std::vector<MeshPrimitiveRange> primitiveRanges;

        for (const MeshPrimitive& primitive : mesh.primitives) 
        {
            size_t oldGlobalVertCount = flatScene.vertices.size();
            size_t oldGlobalIdxCount = flatScene.indices.size();

            flatScene.vertices.reserve(flatScene.vertices.size() + primitive.vertices.size());
            flatScene.vertices.insert(flatScene.vertices.end(), primitive.vertices.begin(), primitive.vertices.end());

            flatScene.indices.reserve(flatScene.indices.size() + primitive.indices.size());
            flatScene.indices.insert(flatScene.indices.end(), primitive.indices.begin(), primitive.indices.end());

            MeshPrimitiveRange range{};
            range.vertexOffset = static_cast<uint32_t>(oldGlobalVertCount);
            range.vertexCount = static_cast<uint32_t>(primitive.vertices.size());
            range.indexOffset = static_cast<uint32_t>(oldGlobalIdxCount);
            range.indexCount = static_cast<uint32_t>(primitive.indices.size());

            primitiveRanges.push_back(range);
        }

        flatScene.meshRanges.push_back(primitiveRanges);
    }

    // Helper to add a mesh instance to the flatenned scene
    void addMeshInstance(const MeshInstance& instance, FlatScene& flatScene)
    {
        size_t oldGeometryCount = flatScene.geometries.size();
        flatScene.instanceFirstGeometry.push_back(static_cast<uint32_t>(oldGeometryCount));
        flatScene.instances.push_back(instance);
        
        bool isLight = glm::length(flatScene.materials.at(instance.materialId).emissionColor) > 0.0;
        const std::vector<MeshPrimitiveRange>& primitiveRanges = flatScene.meshRanges.at(instance.meshIndex);
        for (const MeshPrimitiveRange& range : primitiveRanges)
        {
            gpu::GeometryInfo geomInfo{};
            geomInfo.indexOffset = range.indexOffset;
            geomInfo.vertexOffset = range.vertexOffset;
            geomInfo.materialIndex = instance.materialId;

            flatScene.geometries.push_back(geomInfo);

            if (isLight)
            {
                for (uint32_t i = 0; i < range.indexCount / 3; i++)
                {
                    gpu::EmissiveTri light{};

                    light.v0 = glm::vec3(
                        instance.modelMatrix
                        * glm::vec4(
                            flatScene.vertices.at(
                                geomInfo.vertexOffset
                                + flatScene.indices.at(
                                    geomInfo.indexOffset
                                    + (3 * i)
                                    + 0)).position, 1.0));

                    light.v1 = glm::vec3(
                        instance.modelMatrix
                        * glm::vec4(
                            flatScene.vertices.at(
                                geomInfo.vertexOffset
                                + flatScene.indices.at(
                                    geomInfo.indexOffset
                                    + (3 * i)
                                    + 1)).position, 1.0));

                    light.v2 = glm::vec3(
                        instance.modelMatrix
                        * glm::vec4(
                            flatScene.vertices.at(
                                geomInfo.vertexOffset
                                + flatScene.indices.at(
                                    geomInfo.indexOffset
                                    + (3 * i)
                                    + 2)).position, 1.0));

                    glm::vec3 edge1 = light.v1 - light.v0;
                    glm::vec3 edge2 = light.v2 - light.v0;
                    light.area = glm::length(glm::cross(edge1, edge2)) * 0.5;

                    light.materialIndex = instance.materialId;

                    flatScene.lights.push_back(light);
                }
            }
        }
    }

    // Helper to add a material to the flatenned scene 
    void addMaterial(const Material& material, FlatScene& flatScene)
    {
        // NCHORTEK TODO: This will likely need to change once we're loading
        // real gltf meshes
        gpu::GpuMaterial gpuMat{};
        gpuMat.baseColor = material.color;
        gpuMat.emissionColor = material.color * material.emittance;
        gpuMat.ior = material.indexOfRefraction;
        
        if (material.hasRefractive)
        {
            gpuMat.type = gpu::kSpecGlass;
        }
        else if (material.hasReflective)
        {
            gpuMat.type = gpu::kSpecRefl;
        }
        else
        {
            gpuMat.type = gpu::kDiffuseRefl;
        }

        gpuMat.metallic = 0;
        gpuMat.roughness = 0;
        gpuMat.transmission = 0;

        flatScene.materials.push_back(gpuMat);
    }

    FlatScene flattenScene(const Scene& scene)
    {
        FlatScene flatScene;

        // Populate materials
        for (const Material& mat : scene.materials)
        {
            addMaterial(mat, flatScene);
        }
        
        // Populate mesh data
        for (const MeshData& mesh : scene.meshes)
        {
            addMesh(mesh, flatScene);
        }

        // NOTE: The addMesh loop must come before the addMeshInstance loop,
        // because addMesh populates the primitive ranges that addMeshInstance
        // consumes

        // Populate mesh instance data
        for (const MeshInstance& instance : scene.meshInstances)
        {
            addMeshInstance(instance, flatScene);
        }

        float totalArea = 0;
        for (const gpu::EmissiveTri& light : flatScene.lights)
        {
            totalArea += light.area;
        }

        flatScene.totalLightArea = totalArea;

        if (totalArea > 0)
        {
            float runningAreaSum = 0;
            for (gpu::EmissiveTri& light : flatScene.lights)
            {
                runningAreaSum += light.area;
                light.cdf = runningAreaSum / totalArea;
            }

            flatScene.lights.back().cdf = 1;
        }

        return flatScene;
    }

} // namespace sceneutil