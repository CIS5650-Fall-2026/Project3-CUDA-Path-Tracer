#include "sceneLoader.h"

// Project Includes
#include "scene.h"
#include "bvh.h"
#include "textureLoader.h"

// External Includes
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "tiny_gltf_v3.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// Std Includes
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <filesystem>

using json = nlohmann::json;

static glm::vec3 parseVec3(const json& arrayNode) {
    if (!arrayNode.is_array() || arrayNode.size() < 3) {
        return glm::vec3(0.0f);
    }
    return glm::vec3(arrayNode[0].get<float>(), arrayNode[1].get<float>(), arrayNode[2].get<float>());
}


static std::string getLowercaseExtension(const std::string& filepath)
{
    std::string ext = std::filesystem::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

static glm::mat4 getUpAxisTransform(UpAxis upAxis)
{
    if (upAxis == UpAxis::ZUp) {
        return glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    }
    else if (upAxis == UpAxis::XUp) {
        return glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    }
    else {
        return glm::mat4(1.0f);
    }
}

// Helper to apply both the scale and up axis transform from the import settings to a model
static glm::mat4 getImportTransform(const SceneLoadOptions& options)
{
    return getUpAxisTransform(options.upAxis) * glm::scale(glm::mat4(1.0f), glm::vec3(options.importScale));
}

Material FallbackMaterialSettings::toMaterial() const
{
    Material mat{};
    mat.albedo = albedo;
    mat.albedoTexture = 0;
    mat.emittance = 0.0f;
    mat.ior = ior;

    switch (type) {
    case Type::Mirror:
        mat.type = MaterialType::PerfectSpecular;
        mat.blinnPhong.bRefractive = false;
        break;
    case Type::Glass:
        mat.type = MaterialType::PerfectSpecular;
        mat.blinnPhong.bRefractive = true;
        break;
    case Type::Glossy:
        // Same roughness to exponent mapping as json specular materials
        mat.type = MaterialType::BlinnPhong;
        mat.blinnPhong.specularColor = glm::vec3(1.0f);
        mat.blinnPhong.exponent = glm::mix(1.0f, 1000.0f, 1.0f - roughness);
        mat.blinnPhong.bRefractive = false;
        break;
    case Type::Pbr:
        mat.type = MaterialType::PbrMetallicRoughness;
        mat.pbr.metallic = metallic;
        mat.pbr.roughness = roughness;
        mat.pbr.transmission = 0.0f;
        mat.pbr.metallicRoughnessTexture = 0;
        break;
    default:
        mat.type = MaterialType::OpaqueDiffuse;
        break;
    }

    return mat;
}

static void applyFallbackMaterial(std::vector<Material>& materials, std::vector<Geom>& geometry, const SceneLoadOptions& options)
{
    int fallbackIndex = static_cast<int>(materials.size());
    materials.push_back(options.fallbackMaterial.toMaterial());

    for (Geom& geom : geometry) {
        bool bHasValidMaterial = geom.materialid >= 0 && geom.materialid < fallbackIndex;
        bool bIsLight = bHasValidMaterial && materials[geom.materialid].emittance > 0.0f;

        if (!bHasValidMaterial || (options.bOverrideMaterials && !bIsLight)) {
            geom.materialid = fallbackIndex;
        }
    }
}

std::unique_ptr<Scene> SceneLoader::loadFromFile(const std::string& filepath, const SceneLoadOptions& options)
{
    std::string ext = getLowercaseExtension(filepath);

    if (ext == ".json") {
        return loadFromJson(filepath, options);
    }
    else if (ext == ".gltf" || ext == ".glb") {
        return loadFromGltf(filepath, options);
    }
    else if (ext == ".obj") {
        return loadFromObj(filepath, options);
    }
    else {
        std::cerr << "Unsupported scene file format: " << ext << std::endl;
        return nullptr;
    }
}

std::unique_ptr<Scene> SceneLoader::loadFromJson(const std::string& filepath, const SceneLoadOptions& options)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open scene file: " + filepath);
    }

    json sceneJson;
    try {
        file >> sceneJson;
    }
    catch (const json::parse_error& e) {
        throw std::runtime_error("Failed to parse scene file: " + std::string(e.what()));
    }

    std::vector<Material> materials;
    std::vector<Geom> geometry;
    std::unordered_map<std::string, int> materialMap;

    // Parse materials
    if (sceneJson.contains("Materials")) {
        for (const auto& [name, data] : sceneJson["Materials"].items()) {
            Material mat{};
            if (data.contains("RGB")) {
                mat.albedo = parseVec3(data["RGB"]);
            }
            else {
                mat.albedo = glm::vec3(1.0f);
            }

            std::string type = data.value("TYPE", "Diffuse");
            if (type == "Emitting") {
                mat.emittance = data.value("EMITTANCE", 0.0f);
            }
            else if (type == "Specular") {
                mat.type = MaterialType::BlinnPhong;
                float roughness = data.value("ROUGHNESS", 0.0f);
                if (data.contains("SPECULAR_COLOR")) {
                    mat.blinnPhong.specularColor = parseVec3(data["SPECULAR_COLOR"]);
                }
                else {
                    mat.blinnPhong.specularColor = glm::vec3(1.0f);
                }
                mat.blinnPhong.exponent = glm::mix(1.0f, 1000.0f, 1.0f - roughness);
                mat.blinnPhong.bRefractive = false;
            }
            else if (type == "Mirror") {
                mat.type = MaterialType::PerfectSpecular;
                mat.blinnPhong.bRefractive = false;
            }
            else if (type == "Glass") {
                mat.type = MaterialType::PerfectSpecular;
                mat.blinnPhong.bRefractive = true;
                mat.ior = data.value("IOR", 1.5f);
            }

            materials.push_back(mat);
            materialMap[name] = static_cast<int>(materials.size()) - 1;
        }
    }
    // Parse geometry
    glm::mat4 importTransform = getImportTransform(options);

    if (sceneJson.contains("Objects")) {
        for (const auto& objData : sceneJson["Objects"]) {
            Geom geom{};

            std::string type = objData.value("TYPE", "cube");
            geom.type = (type == "sphere") ? GeomType::SPHERE : GeomType::CUBE;

            std::string matName = objData.value("MATERIAL", "");
            auto it = materialMap.find(matName);
            geom.materialid = (it != materialMap.end()) ? it->second : -1; // -1 gets the fallback material

            // Extract temporary transform components
            glm::vec3 translation = parseVec3(objData["TRANS"]);
            glm::vec3 rotation = parseVec3(objData["ROTAT"]);
            glm::vec3 scale = parseVec3(objData["SCALE"]);

            // Convert to matrices
            glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), translation);
            glm::mat4 rotationMat = glm::mat4(1.0f);
            rotationMat = glm::rotate(rotationMat, glm::radians(rotation.x), glm::vec3(1, 0, 0));
            rotationMat = glm::rotate(rotationMat, glm::radians(rotation.y), glm::vec3(0, 1, 0));
            rotationMat = glm::rotate(rotationMat, glm::radians(rotation.z), glm::vec3(0, 0, 1));
            glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), scale);

            // Set matrices
            geom.transform = importTransform * translationMat * rotationMat * scaleMat;
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::inverseTranspose(geom.transform);

            geometry.push_back(geom);
        }
    }

    applyFallbackMaterial(materials, geometry, options);

    // Sort geometry so lights are first BEFORE building TLAS
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    size_t lightCount = std::distance(geometry.begin(), it);

    // Build TLAS
    std::vector<TLASNode> tlasNodes;

    std::vector<GeometryBVHBuildData> geometryBuildData(geometry.size());
    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
        AABB bounds = AABB::fromGeometry(geom, {});
        geometryBuildData[g].bounds = bounds;
        geometryBuildData[g].centroid = (bounds.min + bounds.max) * 0.5f;
        geometryBuildData[g].geometryIndex = g;
    }

    BVHBuilder::buildTLAS(geometryBuildData, 0, geometryBuildData.size(), tlasNodes);

    std::vector<Triangle> triangles;
    std::vector<BLASNode> blasNodes;

    return std::make_unique<Scene>(geometry, lightCount, triangles, blasNodes, tlasNodes, materials);
}

void parseGltfNodeRecursive(
    const tg3_model& model,
    uint32_t nodeIdx,
    const glm::mat4& parentTransform,
    std::vector<Geom>& outGeometries,
    std::vector<Triangle>& outTriangles)
{
    const tg3_node& node = model.nodes[nodeIdx];

    // Get the local transform of the node
    glm::mat4 localTransform = glm::mat4(1.0f);
    if (node.has_matrix) {
        localTransform = glm::make_mat4(node.matrix);
    }
    else {
        glm::vec3 translation = glm::make_vec3(node.translation);
        glm::quat rotation = glm::make_quat(node.rotation);
        glm::vec3 scale = glm::make_vec3(node.scale);

        localTransform = glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
    }

    // Use parent transform to get global transform
    glm::mat4 globalTransform = parentTransform * localTransform;

    // If the node has a mesh, process it
    if (node.mesh >= 0 && node.mesh < model.meshes_count) {
        const tg3_mesh& mesh = model.meshes[node.mesh];

        // Each primitive will be a geometry struct
        for (uint32_t i = 0; i < mesh.primitives_count; i++) {
            const tg3_primitive& prim = mesh.primitives[i];

            // Get position accessor
            int positionAccessorIdx = -1;
            int uvAccessorIdx = -1;
            for (uint32_t a = 0; a < prim.attributes_count; a++) {
                if (strcmp(prim.attributes[a].key.data, "POSITION") == 0) {
                    positionAccessorIdx = prim.attributes[a].value;
                }
                else if (strcmp(prim.attributes[a].key.data, "TEXCOORD_0") == 0) {
                    uvAccessorIdx = prim.attributes[a].value;
                }
            }

            if (positionAccessorIdx < 0 || positionAccessorIdx >= model.accessors_count) {
                continue;
            }

            // Get vertices in local space
            const tg3_accessor& posAccessor = model.accessors[positionAccessorIdx];
            const tg3_buffer_view& posView = model.buffer_views[posAccessor.buffer_view];
            const tg3_buffer& posBuffer = model.buffers[posView.buffer];

            const uint8_t* posBufferData = posBuffer.data.data + posView.byte_offset + posAccessor.byte_offset;
            uint32_t posStride = posView.byte_stride > 0 ? posView.byte_stride : sizeof(float) * 3;
            uint32_t vertexCount = posAccessor.count;

            const uint8_t* uvBufferData = nullptr;
            uint32_t uvStride = 0;
            int32_t uvComponentType = TG3_COMPONENT_TYPE_FLOAT;
            bool hasUVs = false;

            if (uvAccessorIdx >= 0 && uvAccessorIdx < model.accessors_count) {
                const tg3_accessor& uvAccessor = model.accessors[uvAccessorIdx];
                const tg3_buffer_view& uvView = model.buffer_views[uvAccessor.buffer_view];
                const tg3_buffer& uvBuffer = model.buffers[uvView.buffer];

                // UVs can actually be either floats, unsigned bytes, or unsigned shorts
                uvComponentType = uvAccessor.component_type;
                uint32_t uvSize = 4;
                if (uvComponentType == TG3_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    uvSize = 1;
                }
                else if (uvComponentType == TG3_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    uvSize = 2;
                }

                uvBufferData = uvBuffer.data.data + uvView.byte_offset + uvAccessor.byte_offset;
                uvStride = uvView.byte_stride > 0 ? uvView.byte_stride : uvSize * 2;
                hasUVs = uvAccessor.count == vertexCount;
            }

            // Extract raw local vertices into a temporary vector
            std::vector<glm::vec3> localVertices(vertexCount);
            std::vector<glm::vec2> localUVs(vertexCount, glm::vec2(0.0f));

            for (uint32_t v = 0; v < vertexCount; v++) {
                const float* rawVertexFloats = reinterpret_cast<const float*>(posBufferData + (v * posStride));
                localVertices[v] = glm::vec3(rawVertexFloats[0], rawVertexFloats[1], rawVertexFloats[2]);

                if (hasUVs) {
                    const uint8_t* rawUV = uvBufferData + (v * uvStride);
                    if (uvComponentType == TG3_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        // Convert byte to float by dividing by 255
                        localUVs[v] = glm::vec2(rawUV[0], rawUV[1]) / 255.0f;
                    }
                    else if (uvComponentType == TG3_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        // Convert unsigned short to float by dividing by 65535
                        const uint16_t* rawUVShorts = reinterpret_cast<const uint16_t*>(rawUV);
                        localUVs[v] = glm::vec2(rawUVShorts[0], rawUVShorts[1]) / 65535.0f;
                    }
                    else {
                        // No conversion needed for float types
                        const float* rawUVFloats = reinterpret_cast<const float*>(rawUV);
                        localUVs[v] = glm::vec2(rawUVFloats[0], rawUVFloats[1]);
                    }
                }
            }

            // Assemble triangles
            int startTriangleOffset = static_cast<int>(outTriangles.size());
            int primitiveTriangleCount = 0;

            if (prim.indices >= 0 && static_cast<uint32_t>(prim.indices) < model.accessors_count) {
                // Mesh uses indexing

                // Get index accessor
                const tg3_accessor& idxAccessor = model.accessors[prim.indices];
                const tg3_buffer_view& idxView = model.buffer_views[idxAccessor.buffer_view];
                const tg3_buffer& idxBuffer = model.buffers[idxView.buffer];

                const uint8_t* idxBufferData = idxBuffer.data.data + idxView.byte_offset + idxAccessor.byte_offset;
                uint32_t idxStride = idxView.byte_stride > 0 ? idxView.byte_stride : (idxAccessor.component_type == 5123 ? 2 : 4); // 5123 is GL_UNSIGNED_SHORT

                uint32_t indexCount = idxAccessor.count;
                primitiveTriangleCount = indexCount / 3;

                for (uint32_t idx = 0; idx < indexCount; idx += 3) {
                    uint32_t i0 = 0, i1 = 0, i2 = 0;

                    if (idxAccessor.component_type == 5123) { // GL_UNSIGNED_SHORT
                        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(idxBufferData + (idx * idxStride));
                        i0 = ptr[0]; i1 = ptr[1]; i2 = ptr[2];
                    }
                    else { // GL_UNSIGNED_INT
                        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(idxBufferData + (idx * idxStride));
                        i0 = ptr[0]; i1 = ptr[1]; i2 = ptr[2];
                    }

                    Triangle tri{};
                    tri.v0 = localVertices[i0];
                    tri.v1 = localVertices[i1];
                    tri.v2 = localVertices[i2];
                    tri.uv0 = localUVs[i0];
                    tri.uv1 = localUVs[i1];
                    tri.uv2 = localUVs[i2];
                    outTriangles.push_back(tri);
                }
            }
            else {
                // No indexing, just every 3 vertices are a triangle
                primitiveTriangleCount = vertexCount / 3;
                for (uint32_t v = 0; v < vertexCount; v += 3) {
                    Triangle tri{};
                    tri.v0 = localVertices[v + 0];
                    tri.v1 = localVertices[v + 1];
                    tri.v2 = localVertices[v + 2];
                    tri.uv0 = localUVs[v + 0];
                    tri.uv1 = localUVs[v + 1];
                    tri.uv2 = localUVs[v + 2];
                    outTriangles.push_back(tri);
                }
            }

            Geom geom{};
            geom.type = GeomType::MESH;

            // Grab material index (fallback to material 0)
            geom.materialid = (prim.material >= 0 && prim.material < model.materials_count) ? static_cast<int>(prim.material) : -1; // -1 gets the fallback material

            geom.transform = globalTransform;
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::inverseTranspose(geom.transform);

            geom.triangleOffset = startTriangleOffset;
            geom.triangleCount = primitiveTriangleCount;

            outGeometries.push_back(geom);
        }
    }

    // Recurse to children
    for (uint32_t i = 0; i < node.children_count; i++) {
        parseGltfNodeRecursive(model, node.children[i], globalTransform, outGeometries, outTriangles);
    }
}

// Helper for matching strings since gltf uses its own string struct
static bool gltfStringEquals(const tg3_str& str, const char* text)
{
    size_t length = strlen(text);
    return str.data != nullptr && str.len == length && strncmp(str.data, text, length) == 0;
}

static const tg3_value* findGltfExtension(const tg3_extras_ext& ext, const char* name)
{
    for (uint32_t i = 0; i < ext.extensions_count; i++) {
        if (gltfStringEquals(ext.extensions[i].name, name)) {
            return &ext.extensions[i].value;
        }
    }
    return nullptr;
}

// Try reading a double from a gltf object and return a default value if not found
static double getGltfNumber(const tg3_value* object, const char* key, double defaultValue)
{
    if (object == nullptr || object->type != TG3_VALUE_OBJECT) {
        // Invalid type or no object
        return defaultValue;
    }

    // Search through the objects data until we find one that matches the search
    for (uint32_t i = 0; i < object->object_count; i++) {
        const tg3_kv_pair& pair = object->object_data[i];
        if (!gltfStringEquals(pair.key, key)) {
            continue;
        }

        if (pair.value.type == TG3_VALUE_REAL) {
            return pair.value.real_val;
        }
        if (pair.value.type == TG3_VALUE_INT) {
            return static_cast<double>(pair.value.int_val);
        }
    }
    return defaultValue;
}

void parseGltfMaterials(const tg3_model& model, const std::string& baseDir, std::vector<Material>& outMaterials, std::vector<cudaTextureObject_t>& outTextures, std::vector<cudaArray_t>& outTextureArrays)
{
    // Load all images as textures first
    if (model.images_count > 0) {
        outTextures.resize(model.images_count, 0);
        outTextureArrays.resize(model.images_count, nullptr);

        for (uint32_t i = 0; i < model.images_count; i++) {
            const tg3_image& image = model.images[i];

            if (image.buffer_view >= 0 && image.buffer_view < model.buffer_views_count) {
                // Image is embedded in the binary so load it in a special way
                const tg3_buffer_view& view = model.buffer_views[image.buffer_view];
                const tg3_buffer& buffer = model.buffers[view.buffer];

                cudaTextureDesc texDesc = {};
                texDesc.addressMode[0] = cudaAddressModeWrap;
                texDesc.addressMode[1] = cudaAddressModeWrap;
                texDesc.filterMode = cudaFilterModeLinear;
                texDesc.readMode = cudaReadModeNormalizedFloat;
                texDesc.normalizedCoords = 1;
                if (!TextureLoader::LoadTextureFromMemory(buffer.data.data + view.byte_offset, view.byte_length, outTextures[i], outTextureArrays[i], texDesc)) {
                    std::cerr << "Failed to load embedded image " << i << std::endl;
                }
                continue;
            }

            if (image.uri.data == nullptr) {
                // No uri to look up the texture, so fail
                std::cerr << "Image " << i << " has no uri or buffer view, skipping." << std::endl;
                continue;
            }

            // Load texture from filepath
            std::string uri(image.uri.data, image.uri.len);
            std::string fullImagePath = baseDir + uri;

            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 1;
            if (!TextureLoader::LoadTexture(fullImagePath.c_str(), outTextures[i], outTextureArrays[i], texDesc)) {
                std::cerr << "Failed to load texture: " << fullImagePath << std::endl;
            }
        }
    }

    // Parse materials and assign texture handles
    for (uint32_t i = 0; i < model.materials_count; ++i) {
        const tg3_material& gltfMat = model.materials[i];
        Material mat{};

        mat.type = MaterialType::PbrMetallicRoughness;
        mat.albedoTexture = 0;
        mat.pbr.metallicRoughnessTexture = 0;
        mat.pbr.transmission = 0.0f;
        mat.ior = 1.5f;

        float emissiveSum = gltfMat.emissive_factor[0] + gltfMat.emissive_factor[1] + gltfMat.emissive_factor[2];
        bool bHasEmissiveTexture = gltfMat.emissive_texture.index >= 0;
        if (emissiveSum > 0.0f && bHasEmissiveTexture) {
            // I don't have support for emissive textures right now. Add an extra log to notify this
            std::cerr << "Material " << i << " has textured emission, which isn't supported. Loading it without emission." << std::endl;
        }

        if (emissiveSum > 0.0f && !bHasEmissiveTexture) {
            mat.albedo = glm::vec3(
                gltfMat.emissive_factor[0],
                gltfMat.emissive_factor[1],
                gltfMat.emissive_factor[2]
            );
            mat.emittance = 1.0f;
            outMaterials.push_back(mat);
            continue;
        }

        mat.albedo = glm::vec3(
            gltfMat.pbr_metallic_roughness.base_color_factor[0],
            gltfMat.pbr_metallic_roughness.base_color_factor[1],
            gltfMat.pbr_metallic_roughness.base_color_factor[2]
        );
        mat.emittance = 0.0f;
        mat.pbr.roughness = gltfMat.pbr_metallic_roughness.roughness_factor;
        mat.pbr.metallic = gltfMat.pbr_metallic_roughness.metallic_factor;

        // Loading the transmission extensions since I support transmission in pbr materials
        const tg3_value* transmissionExtension = findGltfExtension(gltfMat.ext, "KHR_materials_transmission");
        mat.pbr.transmission = static_cast<float>(getGltfNumber(transmissionExtension, "transmissionFactor", 0.0));

        const tg3_value* iorExtension = findGltfExtension(gltfMat.ext, "KHR_materials_ior");
        mat.ior = static_cast<float>(getGltfNumber(iorExtension, "ior", 1.5));

        if (mat.pbr.roughness > 0.99f && mat.pbr.metallic == 0.0f && mat.pbr.transmission == 0.0f) {
            mat.type = MaterialType::OpaqueDiffuse;
        }

        if (gltfMat.pbr_metallic_roughness.base_color_texture.index >= 0 &&
            gltfMat.pbr_metallic_roughness.base_color_texture.index < model.textures_count) {
            int texIdx = gltfMat.pbr_metallic_roughness.base_color_texture.index;
            const tg3_texture& textureRef = model.textures[texIdx];

            if (textureRef.source >= 0 && textureRef.source < outTextures.size()) {
                mat.albedoTexture = outTextures[textureRef.source];
            }
            else {
                std::cerr << "Material " << i << " has invalid base color texture index: " << textureRef.source << std::endl;
            }
        }

        if (gltfMat.pbr_metallic_roughness.metallic_roughness_texture.index >= 0 &&
            gltfMat.pbr_metallic_roughness.metallic_roughness_texture.index < model.textures_count) {
            int texIdx = gltfMat.pbr_metallic_roughness.metallic_roughness_texture.index;
            const tg3_texture& textureRef = model.textures[texIdx];

            if (textureRef.source >= 0 && textureRef.source < outTextures.size()) {
                mat.pbr.metallicRoughnessTexture = outTextures[textureRef.source];
            }
            else {
                std::cerr << "Material " << i << " has invalid metallic-roughness texture index: " << textureRef.source << std::endl;
            }
        }

        outMaterials.push_back(mat);
    }

}

std::unique_ptr<Scene> SceneLoader::loadFromGltf(const std::string& filepath, const SceneLoadOptions& options)
{
    std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    tg3_parse_options opts;
    tg3_error_stack errors;
    tg3_model model;

    tg3_parse_options_init(&opts);
    tg3_error_stack_init(&errors);

    tg3_error_code err = tg3_parse_file(&model, &errors, filepath.c_str(), static_cast<uint32_t>(filepath.length()), &opts);

    if (err != TG3_OK) {
        std::string errorsSummary = "";
        for (uint32_t i = 0; i < errors.count; i++) {
            errorsSummary += "\n[TinyGLTF3 Error] " + std::string(errors.entries[i].message);
        }
        tg3_error_stack_free(&errors);
        std::cerr << "Failed to parse glTF file: " << filepath << errorsSummary << std::endl;
        return nullptr;
    }

    std::vector<Material> materials;
    std::vector<Geom> geometry;
    std::vector<Triangle> triangles;
    std::vector<cudaTextureObject_t> textures;
    std::vector<cudaArray_t> textureArrays;

    std::string baseDir = "";
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        baseDir = filepath.substr(0, lastSlash + 1);
    }
    parseGltfMaterials(model, baseDir, materials, textures, textureArrays);

    // Parse geometry from default scene

    glm::mat4 rootTransform = getImportTransform(options);

    uint32_t activeSceneIdx = (model.default_scene >= 0 && model.default_scene < model.scenes_count) ? model.default_scene : 0;
    if (activeSceneIdx < model.scenes_count) {
        const tg3_scene& scene = model.scenes[activeSceneIdx];
        for (uint32_t i = 0; i < scene.nodes_count; ++i) {
            parseGltfNodeRecursive(model, scene.nodes[i], rootTransform, geometry, triangles);
        }
    }

    // Done with parsing
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);

    // Build BLAS
    std::vector<BLASNode> blasNodes;
    std::vector<Triangle> sortedTriangles;

    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];

        if (geom.type != GeomType::MESH || geom.triangleCount <= 0) {
            continue;
        }

        std::vector<TriangleBVHBuildData> triangleBuildData(geom.triangleCount);
        for (int t = 0; t < geom.triangleCount; t++) {
            const Triangle& tri = triangles[geom.triangleOffset + t];
            AABB bounds = AABB::fromTriangle(tri);
            triangleBuildData[t].bounds = bounds;
            triangleBuildData[t].centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
            triangleBuildData[t].triangleIndex = geom.triangleOffset + t;
        }

        std::vector<BLASNode> geomBlasNodes;
        BVHBuilder::buildBLAS(triangleBuildData, 0, triangleBuildData.size(), geomBlasNodes, triangles, sortedTriangles);

        int nodeOffsetStart = blasNodes.size();
        for (size_t n = 0; n < geomBlasNodes.size(); n++) {
            BLASNode& node = geomBlasNodes[n];
            if (node.triangleCount == 0) {
                node.leftChild += nodeOffsetStart;
            }
        }

        geom.blasNodeOffset = nodeOffsetStart;

        blasNodes.insert(blasNodes.end(), geomBlasNodes.begin(), geomBlasNodes.end());
    }

    applyFallbackMaterial(materials, geometry, options);

    // Sort geometry so lights are first BEFORE building TLAS
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    int lightCount = std::distance(geometry.begin(), it);

    // Build TLAS
    std::vector<TLASNode> tlasNodes;

    std::vector<GeometryBVHBuildData> geometryBuildData(geometry.size());
    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
        AABB bounds = AABB::fromGeometry(geom, blasNodes);
        geometryBuildData[g].bounds = bounds;
        geometryBuildData[g].centroid = (bounds.min + bounds.max) * 0.5f;
        geometryBuildData[g].geometryIndex = g;
    }

    BVHBuilder::buildTLAS(geometryBuildData, 0, geometryBuildData.size(), tlasNodes);

    return std::make_unique<Scene>(geometry, lightCount, sortedTriangles, blasNodes, tlasNodes, materials, textures, textureArrays);
}

std::unique_ptr<Scene> SceneLoader::loadFromObj(const std::string& filepath, const SceneLoadOptions& options)
{
    tinyobj::ObjReaderConfig readerConfig;

    std::filesystem::path path(filepath);
    std::string baseDir = path.parent_path().string();
    if (!baseDir.empty()) {
        baseDir += "/";
    }

    readerConfig.mtl_search_path = path.parent_path().string();
    readerConfig.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filepath, readerConfig)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error() << std::endl;
        }
        return nullptr;
    }

    if (!reader.Warning().empty()) {
        std::cerr << "TinyObjReader: " << reader.Warning() << std::endl;
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& objMaterials = reader.GetMaterials();

    std::vector<cudaTextureObject_t> loadedTextures;
    std::vector<cudaArray_t> loadedTextureArrays;
    std::unordered_map<std::string, cudaTextureObject_t> textureCache;

    std::vector<Material> materials;
    std::vector<Geom> geometry;
    std::vector<Triangle> triangles;

    // Parse materials from obj
    for (const auto& objMaterial : objMaterials) {
        Material mat{};

        mat.albedo = glm::vec3(objMaterial.diffuse[0], objMaterial.diffuse[1], objMaterial.diffuse[2]);
        mat.blinnPhong.specularColor = glm::vec3(objMaterial.specular[0], objMaterial.specular[1], objMaterial.specular[2]);
        mat.blinnPhong.exponent = glm::clamp(objMaterial.shininess, 0.0f, 1000.0f);
        mat.ior = glm::max(1.0f, objMaterial.ior);

        mat.albedoTexture = 0;

        if (!objMaterial.diffuse_texname.empty()) {
            std::string fullTexPath = baseDir + objMaterial.diffuse_texname;

            auto iter = textureCache.find(fullTexPath);
            if (iter != textureCache.end()) {
                // Texture already loaded by another material, reuse the index
                mat.albedoTexture = iter->second;
            }
            else {
                // New unique texture, load via your helper class
                cudaTextureObject_t textureObject = 0;
                cudaArray_t textureArray = nullptr;

                cudaTextureDesc texDesc = {};
                texDesc.addressMode[0] = cudaAddressModeWrap;
                texDesc.addressMode[1] = cudaAddressModeWrap;
                texDesc.filterMode = cudaFilterModeLinear;
                texDesc.readMode = cudaReadModeNormalizedFloat;
                texDesc.normalizedCoords = 1;
                if (TextureLoader::LoadTexture(fullTexPath.c_str(), textureObject, textureArray, texDesc)) {
                    loadedTextures.push_back(textureObject);
                    loadedTextureArrays.push_back(textureArray);
                    textureCache[fullTexPath] = textureObject;

                    mat.albedoTexture = textureObject;
                }
            }
        }

        float emissiveSum = objMaterial.emission[0] + objMaterial.emission[1] + objMaterial.emission[2];
        if (emissiveSum > 0.0f) {
            mat.type = MaterialType::OpaqueDiffuse;
            mat.albedo = glm::vec3(objMaterial.emission[0], objMaterial.emission[1], objMaterial.emission[2]);
            mat.emittance = 1.0f;
            materials.push_back(mat);
            continue;
        }

        // Check for perfect glass
        bool bTransparentIllum = objMaterial.illum == 4 || objMaterial.illum == 6 || objMaterial.illum == 7 || objMaterial.illum == 9;
        if (bTransparentIllum || objMaterial.dissolve < 1.0f) {
            mat.type = MaterialType::PerfectSpecular;
            mat.blinnPhong.bRefractive = true;
            if (mat.ior <= 1.0f) {
                mat.ior = 1.5f; // Default IOR for glass if not specified
            }
            materials.push_back(mat);
            continue;
        }

        // Check for perfect mirror
        if (objMaterial.illum == 3) {
            mat.type = MaterialType::PerfectSpecular;
            mat.blinnPhong.bRefractive = false;
            materials.push_back(mat);
            continue;
        }

        bool bSpecular = (mat.blinnPhong.specularColor.r > 0.0f ||
            mat.blinnPhong.specularColor.g > 0.0f ||
            mat.blinnPhong.specularColor.b > 0.0f);

        if (bSpecular && objMaterial.shininess > 0.0f) {
            mat.type = MaterialType::BlinnPhong;
            mat.blinnPhong.bRefractive = false;
            materials.push_back(mat);
            continue;
        }
        else {
            mat.type = MaterialType::OpaqueDiffuse;
        }

        materials.push_back(mat);
    }

    // Flip v coordinate since obj loads images upside down
    for (const auto& shape : shapes) {
        Geom geom{};
        geom.type = GeomType::MESH;

        geom.triangleOffset = triangles.size();
        size_t indexOffset = 0;

        // Iterate over polygons which should be triangles
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);

            Triangle tri{};
            tinyobj::index_t idx0 = shape.mesh.indices[indexOffset + 0];
            tri.v0 = glm::vec3(
                attrib.vertices[3 * size_t(idx0.vertex_index) + 0],
                attrib.vertices[3 * size_t(idx0.vertex_index) + 1],
                attrib.vertices[3 * size_t(idx0.vertex_index) + 2]
            );
            if (idx0.texcoord_index >= 0) {
                tri.uv0 = glm::vec2(
                    attrib.texcoords[2 * size_t(idx0.texcoord_index) + 0],
                    1.0f - attrib.texcoords[2 * size_t(idx0.texcoord_index) + 1]
                );
            }
            else {
                tri.uv0 = glm::vec2(0.0f, 0.0f);
            }

            tinyobj::index_t idx1 = shape.mesh.indices[indexOffset + 1];
            tri.v1 = glm::vec3(
                attrib.vertices[3 * size_t(idx1.vertex_index) + 0],
                attrib.vertices[3 * size_t(idx1.vertex_index) + 1],
                attrib.vertices[3 * size_t(idx1.vertex_index) + 2]
            );
            if (idx1.texcoord_index >= 0) {
                tri.uv1 = glm::vec2(
                    attrib.texcoords[2 * size_t(idx1.texcoord_index) + 0],
                    1.0f - attrib.texcoords[2 * size_t(idx1.texcoord_index) + 1]
                );
            }
            else {
                tri.uv1 = glm::vec2(0.0f, 0.0f);
            }

            tinyobj::index_t idx2 = shape.mesh.indices[indexOffset + 2];
            tri.v2 = glm::vec3(
                attrib.vertices[3 * size_t(idx2.vertex_index) + 0],
                attrib.vertices[3 * size_t(idx2.vertex_index) + 1],
                attrib.vertices[3 * size_t(idx2.vertex_index) + 2]
            );
            if (idx2.texcoord_index >= 0) {
                tri.uv2 = glm::vec2(
                    attrib.texcoords[2 * size_t(idx2.texcoord_index) + 0],
                    1.0f - attrib.texcoords[2 * size_t(idx2.texcoord_index) + 1]
                );
            }
            else {
                tri.uv2 = glm::vec2(0.0f, 0.0f);
            }

            triangles.push_back(tri);

            int objMaterialId = shape.mesh.material_ids[f];
            geom.materialid = (objMaterialId >= 0 && objMaterialId < materials.size()) ? objMaterialId : -1; // -1 gets the fallback material

            indexOffset += fv;
        }

        geom.triangleCount = triangles.size() - geom.triangleOffset;

        geom.transform = getImportTransform(options);
        geom.inverseTransform = glm::inverse(geom.transform);
        geom.invTranspose = glm::inverseTranspose(geom.transform);

        if (geom.triangleCount > 0) {
            geometry.push_back(geom);
        }
    }

    // Build BLAS
    std::vector<BLASNode> blasNodes;
    std::vector<Triangle> sortedTriangles;

    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
        std::vector<TriangleBVHBuildData> triangleBuildData(geom.triangleCount);
        for (int t = 0; t < geom.triangleCount; t++) {
            const Triangle& tri = triangles[geom.triangleOffset + t];
            AABB bounds = AABB::fromTriangle(tri);
            triangleBuildData[t].bounds = bounds;
            triangleBuildData[t].centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
            triangleBuildData[t].triangleIndex = geom.triangleOffset + t;
        }

        geom.triangleOffset = sortedTriangles.size();

        std::vector<BLASNode> geomBlasNodes;
        BVHBuilder::buildBLAS(triangleBuildData, 0, triangleBuildData.size(), geomBlasNodes, triangles, sortedTriangles);

        geom.triangleCount = sortedTriangles.size() - geom.triangleOffset;

        int nodeOffsetStart = blasNodes.size();
        for (size_t n = 0; n < geomBlasNodes.size(); n++) {
            BLASNode& node = geomBlasNodes[n];
            if (node.triangleCount == 0) {
                node.leftChild += nodeOffsetStart;
            }
        }
        geom.blasNodeOffset = nodeOffsetStart;
        blasNodes.insert(blasNodes.end(), geomBlasNodes.begin(), geomBlasNodes.end());
    }

    applyFallbackMaterial(materials, geometry, options);

    // Sort geometry so lights are first BEFORE building TLAS
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    int lightCount = std::distance(geometry.begin(), it);

    // Build TLAS
    std::vector<TLASNode> tlasNodes;

    std::vector<GeometryBVHBuildData> geometryBuildData(geometry.size());
    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
        AABB bounds = AABB::fromGeometry(geom, blasNodes);
        geometryBuildData[g].bounds = bounds;
        geometryBuildData[g].centroid = (bounds.min + bounds.max) * 0.5f;
        geometryBuildData[g].geometryIndex = g;
    }

    BVHBuilder::buildTLAS(geometryBuildData, 0, geometryBuildData.size(), tlasNodes);

    return std::make_unique<Scene>(geometry, lightCount, sortedTriangles, blasNodes, tlasNodes, materials, loadedTextures, loadedTextureArrays);
}