#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#include "tiny_gltf.h"

#include <cfloat>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

template <typename T>
static T readElement(const tinygltf::Model& model, const tinygltf::Accessor& accessor, int index)
{
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const unsigned char* data = model.buffers[view.buffer].data.data()
        + view.byteOffset + accessor.byteOffset + index * accessor.ByteStride(view);
    T value;
    memcpy(&value, data, sizeof(T));
    return value;
}

static int readIndex(const tinygltf::Model& model, const tinygltf::Accessor& accessor, int index)
{
    switch (accessor.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return readElement<unsigned char>(model, accessor, index);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return readElement<unsigned short>(model, accessor, index);
    default:
        return readElement<unsigned int>(model, accessor, index);
    }
}

static glm::mat4 nodeMatrix(const tinygltf::Node& node)
{
    glm::mat4 matrix;
    if (node.matrix.size() == 16)
    {
        for (int i = 0; i < 16; i++)
        {
            matrix[i / 4][i % 4] = (float)node.matrix[i];
        }
        return matrix;
    }
    if (node.translation.size() == 3)
    {
        matrix = glm::translate(matrix, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
    }
    if (node.rotation.size() == 4)
    {
        matrix *= glm::mat4_cast(glm::quat(
            (float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
    }
    if (node.scale.size() == 3)
    {
        matrix = glm::scale(matrix, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
    }
    return matrix;
}

static void loadNode(
    const tinygltf::Model& model,
    int nodeIndex,
    glm::mat4 parentTransform,
    int materialBase,
    bool useFileMaterials,
    std::vector<Triangle>& triangles)
{
    const tinygltf::Node& node = model.nodes[nodeIndex];
    glm::mat4 transform = parentTransform * nodeMatrix(node);
    if (node.mesh >= 0)
    {
        glm::mat3 normalTransform = glm::mat3(glm::inverseTranspose(transform));
        for (const tinygltf::Primitive& primitive : model.meshes[node.mesh].primitives)
        {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
            {
                continue;
            }
            const tinygltf::Accessor& positions = model.accessors[primitive.attributes.at("POSITION")];
            auto normals = primitive.attributes.find("NORMAL");
            auto uvs = primitive.attributes.find("TEXCOORD_0");
            bool hasNormals = normals != primitive.attributes.end();
            bool hasUvs = uvs != primitive.attributes.end()
                && model.accessors[uvs->second].componentType == TINYGLTF_COMPONENT_TYPE_FLOAT;
            bool indexed = primitive.indices >= 0;
            int count = indexed ? model.accessors[primitive.indices].count : positions.count;
            int materialId = materialBase;
            if (useFileMaterials)
            {
                materialId += primitive.material >= 0 ? primitive.material : model.materials.size();
            }

            for (int i = 0; i + 2 < count; i += 3)
            {
                Triangle triangle;
                triangle.materialId = materialId;
                for (int corner = 0; corner < 3; corner++)
                {
                    int vertex = indexed ? readIndex(model, model.accessors[primitive.indices], i + corner) : i + corner;
                    glm::vec3 position = readElement<glm::vec3>(model, positions, vertex);
                    triangle.positions[corner] = glm::vec3(transform * glm::vec4(position, 1.0f));
                    if (hasNormals)
                    {
                        glm::vec3 normal = readElement<glm::vec3>(model, model.accessors[normals->second], vertex);
                        triangle.normals[corner] = glm::normalize(normalTransform * normal);
                    }
                    triangle.uvs[corner] = hasUvs
                        ? readElement<glm::vec2>(model, model.accessors[uvs->second], vertex)
                        : glm::vec2(0.0f);
                }
                if (!hasNormals)
                {
                    glm::vec3 faceNormal = glm::normalize(glm::cross(
                        triangle.positions[1] - triangle.positions[0],
                        triangle.positions[2] - triangle.positions[0]));
                    triangle.normals[0] = triangle.normals[1] = triangle.normals[2] = faceNormal;
                }
                triangles.push_back(triangle);
            }
        }
    }
    for (int child : node.children)
    {
        loadNode(model, child, transform, materialBase, useFileMaterials, triangles);
    }
}

int Scene::addTexture(const unsigned char* pixels, int width, int height, int channels)
{
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.offset = texels.size();
    for (int i = 0; i < width * height; i++)
    {
        const unsigned char* pixel = pixels + i * channels;
        texels.push_back(glm::vec3(pixel[0], pixel[channels > 1 ? 1 : 0], pixel[channels > 2 ? 2 : 0]) / 255.0f);
    }
    textures.push_back(texture);
    return textures.size() - 1;
}

int Scene::loadTexture(const std::string& filename)
{
    int width, height, channels;
    unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (pixels == NULL)
    {
        cout << "Couldn't load " << filename << endl;
        exit(-1);
    }
    int texture = addTexture(pixels, width, height, 3);
    stbi_image_free(pixels);
    return texture;
}

void Scene::loadMesh(const std::string& filename, Geom& geom, bool useFileMaterials, float emittance)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string error;
    std::string warning;
    bool binary = filename.substr(filename.find_last_of('.')) == ".glb";
    bool loaded = binary
        ? loader.LoadBinaryFromFile(&model, &error, &warning, filename)
        : loader.LoadASCIIFromFile(&model, &error, &warning, filename);
    if (!loaded)
    {
        cout << "Couldn't load " << filename << ": " << error << endl;
        exit(-1);
    }

    if (useFileMaterials)
    {
        geom.materialid = materials.size();
        for (const tinygltf::Material& fileMaterial : model.materials)
        {
            Material material{};
            const auto& pbr = fileMaterial.pbrMetallicRoughness;
            material.color = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
            if (pbr.baseColorTexture.index >= 0)
            {
                const tinygltf::Image& image = model.images[model.textures[pbr.baseColorTexture.index].source];
                material.colorTexture = addTexture(image.image.data(), image.width, image.height, image.component);
            }
            if (fileMaterial.normalTexture.index >= 0)
            {
                const tinygltf::Image& image = model.images[model.textures[fileMaterial.normalTexture.index].source];
                material.bumpTexture = addTexture(image.image.data(), image.width, image.height, image.component);
                material.normalMap = true;
            }
            if (fileMaterial.emissiveTexture.index >= 0)
            {
                const tinygltf::Image& image = model.images[model.textures[fileMaterial.emissiveTexture.index].source];
                material.emissiveTexture = addTexture(image.image.data(), image.width, image.height, image.component);
                material.emittance = emittance;
            }
            if (fileMaterial.extensions.count("KHR_materials_transmission"))
            {
                material.specular.color = material.color;
                material.hasRefractive = 1.0f;
                material.indexOfRefraction = 1.5f;
            }
            materials.push_back(material);
        }
        Material defaultMaterial{};
        defaultMaterial.color = glm::vec3(0.8f);
        materials.push_back(defaultMaterial);
    }

    geom.triangleStart = triangles.size();
    const tinygltf::Scene& gltfScene = model.scenes[model.defaultScene >= 0 ? model.defaultScene : 0];
    for (int node : gltfScene.nodes)
    {
        loadNode(model, node, geom.transform, geom.materialid, useFileMaterials, triangles);
    }
    geom.triangleCount = triangles.size() - geom.triangleStart;

    geom.boundsMin = glm::vec3(FLT_MAX);
    geom.boundsMax = glm::vec3(-FLT_MAX);
    for (int i = geom.triangleStart; i < triangles.size(); i++)
    {
        for (const glm::vec3& position : triangles[i].positions)
        {
            geom.boundsMin = glm::min(geom.boundsMin, position);
            geom.boundsMax = glm::max(geom.boundsMax, position);
        }
    }
    geom.bvhRoot = bvh.build(triangles, geom.triangleStart, triangles.size());

    cout << "Loaded " << filename << ": " << geom.triangleCount << " triangles, "
        << bvh.nodes.size() << " BVH nodes, bounds " << glm::to_string(geom.boundsMin)
        << " to " << glm::to_string(geom.boundsMax) << endl;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    std::string sceneDirectory = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Subsurface")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.scatterDistance = p["SCATTER_DIST"];
            newMaterial.indexOfRefraction = p.value("IOR", 1.3f);
        }
        if (p.contains("TEXTURE"))
        {
            std::string texture = p["TEXTURE"];
            if (texture == "checker")
            {
                newMaterial.texture = CHECKER;
            }
            else if (texture == "marble")
            {
                newMaterial.texture = MARBLE;
            }
            else
            {
                newMaterial.colorTexture = loadTexture(sceneDirectory + texture);
            }
        }
        if (p.contains("BUMP_MAP"))
        {
            newMaterial.bumpTexture = loadTexture(sceneDirectory + p["BUMP_MAP"].get<std::string>());
            newMaterial.bumpScale = p.value("BUMP_SCALE", 1.0f);
        }
        newMaterial.textureScale = p.value("TEXTURE_SCALE", 1.0f);
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    bvh.maxDepth = glm::min((int)data.value("BVH_MAX_DEPTH", 32), 60);
    bvh.leafSize = data.value("BVH_LEAF_SIZE", 4);
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mandelbulb")
        {
            newGeom.type = MANDELBULB;
        }
        else if (type == "menger")
        {
            newGeom.type = MENGER;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        bool hasMaterial = p.contains("MATERIAL");
        newGeom.materialid = hasMaterial ? MatNameToID[p["MATERIAL"]] : 0;
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        newGeom.velocity = glm::vec3(0.0f);
        if (p.contains("VELOCITY"))
        {
            const auto& velocity = p["VELOCITY"];
            newGeom.velocity = glm::vec3(velocity[0], velocity[1], velocity[2]);
        }
        if (newGeom.type == MESH)
        {
            loadMesh(sceneDirectory + p["FILE"].get<std::string>(), newGeom, !hasMaterial, p.value("EMITTANCE", 1.0f));
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DIST", 1.0f);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    state.albedo = state.image;
    state.normal = state.image;
    state.denoised = state.image;
}
