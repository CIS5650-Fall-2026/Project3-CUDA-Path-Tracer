#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <cfloat>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include "stb_image.h"
#include <cmath>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        buildBVH();
        computeSceneRadius();
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

//compute our bounding sphere for sending out our shadow ray for our directional light
void Scene::computeSceneRadius()
{
    if (bvhNodes.empty()) {
        sceneRadius = 0.0f;
        return;
    }

    const BVHNode& root = bvhNodes[0];
    sceneRadius = 0.5f * glm::length(root.maxCorner - root.minCorner);
}

static void getPrimitiveBounds(
    const PrimitiveRef& primitive,
    const std::vector<Triangle>& triangles,
    const std::vector<Geom>& geoms,
    glm::vec3& minCorner,
    glm::vec3& maxCorner)
{
    minCorner = glm::vec3(FLT_MAX);
    maxCorner = glm::vec3(-FLT_MAX);

    //for triangles
    if (primitive.type == PRIMITIVE_TRIANGLE) {
        const Triangle& triangle = triangles[primitive.index];

        for (int i = 0; i < 3; ++i) {
            minCorner = glm::min(minCorner, triangle.positions[i]);
            maxCorner = glm::max(maxCorner, triangle.positions[i]);
        }
    }
    //for other primtives
    else {
        const Geom& geom = geoms[primitive.index];

        //transform from local to global space
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    glm::vec3 localCorner(
                        x == 0 ? -0.5f : 0.5f,
                        y == 0 ? -0.5f : 0.5f,
                        z == 0 ? -0.5f : 0.5f
                    );

                    glm::vec3 worldCorner = glm::vec3( geom.transform * glm::vec4(localCorner, 1.0f));

                    minCorner = glm::min(minCorner, worldCorner);
                    maxCorner = glm::max(maxCorner, worldCorner);
                }
            }
        }
    }
}

//building our bvh structures on CPU and clearing our past one
void Scene::buildBVH()
{
    bvhNodes.clear();
    bvhPrimitives.clear();

    //fill in our indices for triangles
    for (size_t i = 0; i < triangles.size(); i++) {
        bvhPrimitives.push_back({PRIMITIVE_TRIANGLE, static_cast<int>(i) });
    }

    //fill in for other geom types
    for (size_t i = 0; i < geoms.size(); ++i) {
        bvhPrimitives.push_back({PRIMITIVE_GEOM, static_cast<int>(i)});
    }

    //fill in our node list
    if (!bvhPrimitives.empty()) {
        buildBVHNode(0, static_cast<int>(bvhPrimitives.size()));
    }
}

int Scene::buildBVHNode(int first, int count)
{
    BVHNode node{};
    node.minCorner = glm::vec3(FLT_MAX);
    node.maxCorner = glm::vec3(-FLT_MAX);

    for (int offset = 0; offset < count; offset++) {
        glm::vec3 primitiveMin;
        glm::vec3 primitiveMax;

        getPrimitiveBounds(
            bvhPrimitives[first + offset],
            triangles,
            geoms,
            primitiveMin,
            primitiveMax
        );

        node.minCorner = glm::min(node.minCorner, primitiveMin);
        node.maxCorner = glm::max(node.maxCorner, primitiveMax);
    }

    int nodeIndex = static_cast<int>(bvhNodes.size());
    bvhNodes.push_back(node);
    
    if (count <= 1) {
        bvhNodes[nodeIndex].firstPrimitive = first;
        bvhNodes[nodeIndex].primitiveCount = count;
        return nodeIndex;
    }

    //choose the longest axis of this nodes bounding box
    glm::vec3 extent = node.maxCorner - node.minCorner;

    int axis = 0;
    if (extent.y > extent[axis]) 
        axis = 1;
    
    if (extent.z > extent[axis]) 
        axis = 2;
    

    //divide the indices by their center along that axis
    int middle = first + count / 2;

    //rearranges our prim references into 2 groups around the middle position
    std::nth_element(
        bvhPrimitives.begin() + first,
        bvhPrimitives.begin() + middle,
        bvhPrimitives.begin() + first + count,
        [this, axis](const PrimitiveRef& a, const PrimitiveRef& b) {
            glm::vec3 minA, maxA;
            glm::vec3 minB, maxB;

            getPrimitiveBounds(a, triangles, geoms, minA, maxA);
            getPrimitiveBounds(b, triangles, geoms, minB, maxB);

            float centerA = 0.5f * (minA[axis] + maxA[axis]);
            float centerB = 0.5f * (minB[axis] + maxB[axis]);

            return centerA < centerB;
        }
    );

    //building both halves recursively
    int leftChild = buildBVHNode(first, middle - first);
    int rightChild = buildBVHNode(middle, first + count - middle);

    bvhNodes[nodeIndex].leftChild = leftChild;
    bvhNodes[nodeIndex].rightChild = rightChild;

    //internal nodes keep triangleCount==0
    return nodeIndex;
}

//get our node transformations for our node relative to its parent
static glm::mat4 getNodeTransform(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16) {
        glm::mat4 result(1.0f);

        for (int column = 0; column < 4; column++) {
            for (int row = 0; row < 4; row++) {
                result[column][row] =
                    static_cast<float>(node.matrix[column * 4 + row]);
            }
        }

        return result;
    }

    glm::vec3 translation(0.0f);
    glm::vec3 scale(1.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

    if (node.translation.size() == 3) {
        translation = glm::vec3(
            node.translation[0],
            node.translation[1],
            node.translation[2]
        );
    }

    if (node.scale.size() == 3) {
        scale = glm::vec3(
            node.scale[0],
            node.scale[1],
            node.scale[2]
        );
    }

    if (node.rotation.size() == 4) {
        //we do this because gltf stores x, y, z, w while GLM's constructor takes w, x, y, z.
        rotation = glm::quat(
            node.rotation[3],
            node.rotation[0],
            node.rotation[1],
            node.rotation[2]
        );
    }

    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

float srgbToLinear(float value)
{
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    else {
        return std::pow((value + 0.055f) / 1.055f, 2.4f);
    }
}

int Scene::loadTexture(const std::string& filename, bool isColor)
{
    int width;
    int height;
    int channels;

    unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &channels, 3);

    Texture texture{};
    texture.width = width;
    texture.height = height;
    texture.pixelOffset = static_cast<int>(texturePixels.size());

    size_t pixelCount = static_cast<size_t>(width) * height;

    for (size_t i = 0; i < pixelCount; i++) {
        glm::vec3 color(pixels[3 * i] / 255.0f, pixels[3 * i + 1] / 255.0f, pixels[3 * i + 2] / 255.0f);

        if (isColor) {
            color = glm::vec3(srgbToLinear(color.r), srgbToLinear(color.g), srgbToLinear(color.b));
        }

        texturePixels.push_back(color);
    }

    stbi_image_free(pixels);

    int textureId = static_cast<int>(textures.size());
    textures.push_back(texture);

    return textureId;
}

//take our gltf file and create our indeces and vertices
void Scene::loadGLTF(const std::string& filename, const glm::mat4& transform, bool receiveAccentLights, float atmosphereStrength)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;

    std::string error;
    std::string warning;

    std::string extension = filename.substr(filename.find_last_of('.'));

    bool loaded = false;

    //loading in the file
    if (extension == ".glb") {
        loaded = loader.LoadBinaryFromFile(&model, &error, &warning, filename);
    }
    else if (extension == ".gltf") {
        loaded = loader.LoadASCIIFromFile(&model, &error, &warning, filename);
    }
    else {
        return;
    }

    if (!loaded) {
        return;
    }

    //map this files image indices to our scenes texture indices
    std::vector<int> imageToTexture(model.images.size(), -1);

    for (size_t imageIndex = 0; imageIndex < model.images.size(); imageIndex++) {

        const auto& image = model.images[imageIndex];

        if (image.width <= 0 || 
            image.height <= 0 ||
            image.bits != 8 ||
            image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE ||
            image.component < 1 || image.component > 4) {

            continue;

        }

        size_t pixelCount = static_cast<size_t>(image.width) * image.height;

        if (image.image.size() < pixelCount * static_cast<size_t>(image.component)) {

            continue;

        }

        Texture texture{};
        texture.width = image.width;
        texture.height = image.height;
        texture.pixelOffset = static_cast<int>(texturePixels.size());

        for (size_t pixel = 0; pixel < pixelCount; pixel++) {
            size_t offset = pixel * image.component;

            float r = image.image[offset] / 255.0f;
            float g = r;
            float b = r;

            //one or two channels mean grayscale or grayscale and alpha
            //while three or four channels mean rgb or rgba.
            if (image.component >= 3) {
                g = image.image[offset + 1] / 255.0f;
                b = image.image[offset + 2] / 255.0f;
            }

            texturePixels.emplace_back(
                srgbToLinear(r),
                srgbToLinear(g),
                srgbToLinear(b));
        }

        imageToTexture[imageIndex] = static_cast<int>(textures.size());

        textures.push_back(texture);
    }

    Material fallbackMaterial{};
    fallbackMaterial.receiveAccentLights = receiveAccentLights;
    fallbackMaterial.atmosphereStrength = atmosphereStrength;
    fallbackMaterial.color = glm::vec3(0.2f, 0.5f, 0.9f);

    int fallbackMaterialId = static_cast<int>(materials.size());
    materials.push_back(fallbackMaterial);

    

    std::vector<int> materialMap(model.materials.size(), -1);

    for (size_t i = 0; i < model.materials.size(); i++) {
        const auto& source = model.materials[i];

        Material material{};
        material.receiveAccentLights = receiveAccentLights;
        material.atmosphereStrength = atmosphereStrength;
        const auto& factor = source.pbrMetallicRoughness.baseColorFactor;

        material.color = glm::vec3(factor[0], factor[1], factor[2]);

        int textureIndex = source.pbrMetallicRoughness.baseColorTexture.index;
        int texCoord = source.pbrMetallicRoughness.baseColorTexture.texCoord;

        //for the gltf I use they have the colors attatched to emmision instead of having a base color
        //I just swap it so that their emissive textures are the surface colors
        if (source.emissiveTexture.index >= 0) {
            textureIndex = source.emissiveTexture.index;
            texCoord = source.emissiveTexture.texCoord;
            material.color = glm::vec3(1.0f);
        }

        if (textureIndex >= 0 && textureIndex < static_cast<int>(model.textures.size())) {

            int imageIndex = model.textures[textureIndex].source;

            if (texCoord != 0) {
                std::cerr << "Only TEXCOORD_0 is supported" << std::endl;
            }
            else if (imageIndex >= 0 &&
                imageIndex < static_cast<int>(imageToTexture.size())) {
                material.colorTextureId = imageToTexture[imageIndex];
            }
        }

        materialMap[i] = static_cast<int>(materials.size());
        materials.push_back(material);
    }

    if (model.scenes.empty())
        return;

    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    const auto& gltfScene = model.scenes.at(sceneIndex);

    struct PendingNode {
        int nodeIndex;
        glm::mat4 parentTransform;
    };

    std::vector<PendingNode> pending;

    for (int rootIndex : gltfScene.nodes) {
        pending.push_back({ rootIndex, transform });
    }

    while (!pending.empty()) {
        PendingNode current = pending.back();
        pending.pop_back();

        const auto& node = model.nodes.at(current.nodeIndex);

        //for the gltf im loading it has an outline that messes up stuff in the color. I just skip it

        if (node.name == "outline_2") {
            continue;
        }

        glm::mat4 worldTransform =
            current.parentTransform * getNodeTransform(node);

        //children inherit this nodes accumulated transform
        for (int childIndex : node.children) {
            pending.push_back({ childIndex, worldTransform });
        }

        //a node may only group children, without containing a mesh
        if (node.mesh < 0) {
            continue;
        }

        const auto& mesh = model.meshes.at(node.mesh);

        for (const auto& primitive : mesh.primitives) {
            auto positionEntry = primitive.attributes.find("POSITION");

            if (positionEntry == primitive.attributes.end()) {
                std::cerr << "Primitive has no positions." << std::endl;
                continue;
            }

            int accessorIndex = positionEntry->second;
            const auto& accessor = model.accessors.at(accessorIndex);

            if (accessor.bufferView < 0 ||
                accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
                accessor.type != TINYGLTF_TYPE_VEC3 ||
                accessor.sparse.isSparse) {

                std::cerr << "Unsupported position layout." << std::endl;
                continue;
            }

            const auto& view = model.bufferViews.at(accessor.bufferView);
            const auto& buffer = model.buffers.at(view.buffer);

            size_t startOffset = view.byteOffset + accessor.byteOffset;
            int stride = accessor.ByteStride(view);

            if (stride < 12) {
                std::cerr << "Invalid position stride." << std::endl;
                continue;
            }

            std::vector<glm::vec3> positions;

            //creating our vertices list
            for (size_t i = 0; i < accessor.count; i++) {
                size_t offset = startOffset + i * static_cast<size_t>(stride);

                float xyz[3];
                std::memcpy(xyz, buffer.data.data() + offset, sizeof(xyz));

                positions.emplace_back(xyz[0], xyz[1], xyz[2]);
            }

            std::vector<glm::vec3> normals;

            auto normalEntry = primitive.attributes.find("NORMAL");

            //getting our normals per vertex
            if (normalEntry != primitive.attributes.end()) {
                const auto& normalAccessor =
                    model.accessors.at(normalEntry->second);

                if (normalAccessor.bufferView < 0 ||
                    normalAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
                    normalAccessor.type != TINYGLTF_TYPE_VEC3 ||
                    normalAccessor.sparse.isSparse ||
                    normalAccessor.count != positions.size()) {

                    std::cerr << "Unsupported normal layout" << std::endl;
                    continue;

                }

                const auto& normalView = model.bufferViews.at(normalAccessor.bufferView);
                const auto& normalBuffer = model.buffers.at(normalView.buffer);

                size_t normalStart = normalView.byteOffset + normalAccessor.byteOffset;
                int normalStride = normalAccessor.ByteStride(normalView);

                if (normalStride < 12) {
                    std::cerr << "Invalid normal stride" << std::endl;
                    continue;
                }

                glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(worldTransform)));

                for (size_t i = 0; i < normalAccessor.count; i++) {
                    size_t offset = normalStart + i * static_cast<size_t>(normalStride);

                    float xyz[3];

                    if (offset > normalBuffer.data.size() || sizeof(xyz) > normalBuffer.data.size() - offset) {
                        
                        std::cerr << "Normal data out of bounds" << std::endl;
                        return;

                    }

                    std::memcpy(xyz, normalBuffer.data.data() + offset, sizeof(xyz));

                    glm::vec3 normal = normalMatrix * glm::vec3(xyz[0], xyz[1], xyz[2]);

                    float length = glm::length(normal);
                    normals.push_back(length > 0.0f ? normal / length : glm::vec3(0.0f));
                }
            }

            //uv!
            std::vector<glm::vec2> uvs;

            auto uvEntry = primitive.attributes.find("TEXCOORD_0");

            if (uvEntry != primitive.attributes.end()) {
                const auto& uvAccessor = model.accessors.at(uvEntry->second);

                //just am trying to get my models going, so we use flat uv's
                if (uvAccessor.bufferView < 0 ||
                    uvAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
                    uvAccessor.type != TINYGLTF_TYPE_VEC2 ||
                    uvAccessor.sparse.isSparse ||
                    uvAccessor.count != positions.size()) {

                    std::cerr << "Unsupported UV layout" << std::endl;
                    continue;

                }

                const auto& uvView = model.bufferViews.at(uvAccessor.bufferView);
                const auto& uvBuffer = model.buffers.at(uvView.buffer);

                size_t uvStart = uvView.byteOffset + uvAccessor.byteOffset;
                int uvStride = uvAccessor.ByteStride(uvView);

                if (uvStride < 8) {
                    std::cerr << "Invalid UV stride" << std::endl;
                    continue;
                }

                for (size_t i = 0; i < uvAccessor.count; i++) {
                    size_t offset = uvStart + i * static_cast<size_t>(uvStride);

                    float coordinates[2];

                    if (offset > uvBuffer.data.size() || sizeof(coordinates) > uvBuffer.data.size() - offset) {
                        std::cerr << "UV data out of bounds" << std::endl;
                        return;
                    }

                    std::memcpy(coordinates, uvBuffer.data.data() + offset, sizeof(coordinates));

                    uvs.emplace_back(coordinates[0], coordinates[1]);
                }
            }

            if (primitive.indices < 0) {
                std::cerr << "This first loader requires indices." << std::endl;
                continue;
            }

            const auto& indexAccessor = model.accessors.at(primitive.indices);

            if (indexAccessor.bufferView < 0 ||
                indexAccessor.type != TINYGLTF_TYPE_SCALAR ||
                (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT && indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) ||
                indexAccessor.sparse.isSparse)

            {
                std::cerr << "Unsupported index layout." << std::endl;
                continue;
            }

            bool uses32BitIndices = indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;

            size_t indexBytes = uses32BitIndices ? sizeof(std::uint32_t) : sizeof(std::uint16_t);

            const auto& indexView = model.bufferViews.at(indexAccessor.bufferView);
            const auto& indexBuffer = model.buffers.at(indexView.buffer);

            size_t indexStart = indexView.byteOffset + indexAccessor.byteOffset;

            std::vector<unsigned int> indices;

            //create our index buffer
            for (size_t i = 0; i < indexAccessor.count; i++) {
                size_t offset = indexStart + i * indexBytes;

                if (offset > indexBuffer.data.size() || indexBytes > indexBuffer.data.size() - offset)
                    return;

                std::uint32_t vertexIndex = 0;

                //added for my lucy model, used to be just 16 bit
                if (uses32BitIndices) {
                    std::memcpy(
                        &vertexIndex,
                        indexBuffer.data.data() + offset,
                        sizeof(vertexIndex)
                    );
                }
                else {
                    std::uint16_t smallIndex;

                    std::memcpy(
                        &smallIndex,
                        indexBuffer.data.data() + offset,
                        sizeof(smallIndex)
                    );

                    vertexIndex = smallIndex;
                }

                if (vertexIndex >= positions.size())
                    return;

                indices.push_back(vertexIndex);
            }

            // This version only assembles independent triangles
            if (primitive.mode != -1 &&
                primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cerr << "Unsupported primitive mode." << std::endl;
                continue;
            }

            if (indices.size() % 3 != 0) {
                std::cerr << "Triangle index count must be divisible by 3."
                    << std::endl;
                continue;
            }

            for (size_t i = 0; i < indices.size(); i += 3) {
                Triangle triangle{};

                for (int corner = 0; corner < 3; corner++) {
                    unsigned int vertexIndex = indices[i + corner];

                    triangle.positions[corner] = glm::vec3(worldTransform * glm::vec4(positions[vertexIndex], 1.0f));

                    if (!uvs.empty()) {
                        triangle.uvs[corner] = uvs[vertexIndex];
                    }
                    else {
                        triangle.uvs[corner] = glm::vec2(0.0f);
                    }
                }

                glm::vec3 edgeAB =
                    triangle.positions[1] - triangle.positions[0];
                glm::vec3 edgeAC =
                    triangle.positions[2] - triangle.positions[0];

                glm::vec3 faceNormal = glm::cross(edgeAB, edgeAC);
                float normalLength = glm::length(faceNormal);

                if (normalLength <= 0.0f) {
                    continue; // Skip triangles with zero area.
                }

                faceNormal /= normalLength;

                for (int corner = 0; corner < 3; corner++) {
                    unsigned int vertexIndex = indices[i + corner];

                    //incase our model doesn't have a valid normal, do flat shading
                    glm::vec3 normal = faceNormal;

                    if (!normals.empty() &&
                        glm::dot(normals[vertexIndex], normals[vertexIndex]) > 0.0f) {
                        normal = normals[vertexIndex];
                    }

                    triangle.normals[corner] = normal;
                }

                triangle.materialId = fallbackMaterialId;

                //now triangles that belong to a primitive will now receive that prims mat
                if (primitive.material >= 0 &&
                    primitive.material < static_cast<int>(materialMap.size())) {
                    triangle.materialId = materialMap[primitive.material];
                }

                triangles.push_back(triangle);
            }
        }
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    const auto& lightsData = data["DistantLights"];
    for (const auto& p : lightsData) {
        DistantLight light{};

        const auto& direction = p["TO_LIGHT"];
        const auto& strength = p["STRENGTH"];

        light.toLight = glm::vec3(direction[0], direction[1], direction[2]);

        light.strength = glm::vec3(strength[0], strength[1], strength[2]);

        light.toLight = glm::normalize(light.toLight);

        distantLights.push_back(light);
    }


    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
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
            newMaterial.hasReflective = 1.0f;
        }
        if (p.contains("COLOR_TEXTURE")) {
            std::filesystem::path path = p["COLOR_TEXTURE"].get<std::string>();

            if (path.is_relative()) {
                path = std::filesystem::path(jsonName).parent_path() / path;
            }

            newMaterial.colorTextureId = loadTexture(path.string(), true);
        }
        if (p.contains("BUMP_TEXTURE")) {
            std::filesystem::path path = p["BUMP_TEXTURE"].get<std::string>();

            if (path.is_relative()) {
                path = std::filesystem::path(jsonName).parent_path() / path;
            }

            newMaterial.bumpTextureId = loadTexture(path.string(), false);
        }

        newMaterial.receiveAccentLights = p.value("RECEIVE_ACCENT_LIGHTS", true);

        //default bump value
        newMaterial.bumpStrength = p.value("BUMP_STRENGTH", 0.02f);

        MatNameToID[name] = materials.size();

        newMaterial.useCheckerboard = p.value("CHECKERBOARD", false);
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

        if (type == "mesh") {
            std::filesystem::path modelPath = p.at("FILE").get<std::string>();

            //here we resolve relative model paths from the scene JSON's folder.
            if (modelPath.is_relative()) {
                modelPath = std::filesystem::path(jsonName).parent_path() / modelPath;
            }

            Geom newGeom;
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);

            const size_t firstMaterial = materials.size();

            loadGLTF(modelPath.string(), newGeom.transform, p.value("RECEIVE_ACCENT_LIGHTS", true), p.value("ATMOSPHERE_STRENGTH", 0.0f));


            const float brightness = p.value("LIGHTING_MULTIPLIER", 1.0f);
            for (size_t i = firstMaterial; i < materials.size(); ++i) {
                materials[i].lightingMultiplier = brightness;
            }
            continue;
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
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
}
