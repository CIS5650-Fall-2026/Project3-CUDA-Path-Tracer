#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "json.hpp"
#include <stb_image.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "tiny_gltf.h"

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
        return;
    }
    else if (ext == ".gltf" || ext == ".glb") {
        // wrapper has camera setup etc
        loadFromGltfWrapper(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

struct CompareByAxis
{
    int axis;
    CompareByAxis(int a) : axis(a) {}

    bool operator()(const Triangle& a, const Triangle& b) const
    {
        float ca = (a.v0[axis] + a.v1[axis] + a.v2[axis]) / 3.0f;
        float cb = (b.v0[axis] + b.v1[axis] + b.v2[axis]) / 3.0f;
        return ca < cb;
    }
};

void buildTree(std::vector<Triangle>& triangleArray, std::vector<node>& nodes, int triStart, int triEnd) {
    node currentNode;
    // calculate min, max
    glm::vec3 minBound(1000000000.0f, 1000000000.0f, 1000000000.0f);
    glm::vec3 maxBound(-1000000000.0f, -1000000000.0f, -1000000000.0f);
    for (int i = triStart; i < triEnd; i++) {
        Triangle tri = triangleArray[i];
        minBound = glm::min(minBound, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
        maxBound = glm::max(maxBound, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
    }
    currentNode.minBound = minBound;
    currentNode.maxBound = maxBound;

    int count = triEnd - triStart;
    // too small for children nodes
    if (count <= 4) {
        currentNode.leftChild = -1;
        currentNode.rightChild = -1;
        currentNode.triStart = triStart;
        currentNode.triCount = count;
        nodes.push_back(currentNode);
        return;
    }
    
    // split on longer axis
    glm::vec3 temp = maxBound - minBound;
    int axis = 0;
    if (temp.y > temp.x) {
        axis = 1;
    }
    if (temp.z > temp[axis]) {
        axis = 2;
    }
    // need for spacial boxes
    std::sort(
        triangleArray.begin() + triStart,
        triangleArray.begin() + triEnd,
        CompareByAxis(axis)
    );

    int thisIndex = (int)nodes.size();
    nodes.push_back(currentNode);

    int mid = triStart + count / 2;
    int leftIndex = (int)nodes.size();
    buildTree(triangleArray, nodes, triStart, mid);

    int rightIndex = (int)nodes.size();
    buildTree(triangleArray, nodes, mid, triEnd);

    nodes[thisIndex].leftChild = leftIndex;
    nodes[thisIndex].rightChild = rightIndex;
    nodes[thisIndex].triStart = -1;
    nodes[thisIndex].triCount = 0;
}

bool myImageLoader(tinygltf::Image* image, const int image_idx, std::string* err,
    std::string* warn, int req_width, int req_height,
    const unsigned char* bytes, int size, void* user_data)
{
    int w, h, comp;
    unsigned char* decoded = stbi_load_from_memory(bytes, size, &w, &h, &comp, 0);
    if (!decoded) {
        if (err) *err += "Failed to decode image\n";
        return false;
    }
    image->width = w;
    image->height = h;
    image->component = comp;
    image->bits = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    image->image.assign(decoded, decoded + (w * h * comp));
    stbi_image_free(decoded);
    return true;
}

std::vector<Triangle> loadFromGltf(
    const std::string& path, 
    std::vector<Material>& materials,
    std::vector<node>& nodes,
    std::vector<Texture>& textures,
    std::vector<unsigned char>& textureData
) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    loader.SetImageLoader(myImageLoader, nullptr);
    bool ok = false;

    if (path.substr(path.find_last_of('.')) == ".gltf") {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }
    else {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    }

    if (!ok) {
        fprintf(stderr, "%s\n", err.c_str());
    }
    if (!warn.empty()) {
        fprintf(stderr, "%s\n", warn.c_str());
    }

    std::unordered_map<int, int> materialIdMap;

    std::vector<Triangle> triangleArray;
    int maxMesh = -1; // mesh cap or it might run too slow
    for (const auto& mesh : model.meshes) {
        maxMesh--;
        if (maxMesh == 0) {
            break;
        }
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode == TINYGLTF_MODE_TRIANGLES || primitive.mode == -1) {
                // is a triangle
                if (primitive.indices >= 0) {
                    // for getting indices
                    const tinygltf::Accessor& idxAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& idxBufferView = model.bufferViews[idxAccessor.bufferView];
                    const tinygltf::Buffer& idxBuffer = model.buffers[idxBufferView.buffer];
                    size_t compSize =
                        (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) ? 1 :
                        (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) ? 2 : 4;
                    size_t stride = idxBufferView.byteStride ? idxBufferView.byteStride : compSize;

                    std::vector<int> indices;
                    for (int i = 0; i < idxAccessor.count; i++) {
                        size_t offset = idxBufferView.byteOffset + idxAccessor.byteOffset + i * stride;
                        const unsigned char* p = &idxBuffer.data[offset];

                        int value;
                        if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                            value = *p;
                        } else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                            value = *reinterpret_cast<const uint16_t*>(p);
                        } else {
                            value = *reinterpret_cast<const uint32_t*>(p);
                        }
                        indices.push_back(value);
                    }

                    // custom options for some meshes
                    glm::vec3 meshScale(1.0f);
                    glm::vec3 meshTranslate(0.0f);
                    float meshRotate = 0.0f;
                    glm::vec3 meshRotateDir(1, 0, 0);
                    if (path == "scenes/shiba/scene.gltf") {
                        meshScale = glm::vec3(5.0f);
                        meshTranslate = glm::vec3(0.0f, 5.0f, 0.0f);
                        meshRotate = -90.0f;
                    }
                    else if (path == "scenes/the_billiards_room/scene.gltf") {
                        meshScale = glm::vec3(1.0f);
                        meshTranslate = glm::vec3(0.0f, 0.0f, 0.0f);
                        meshRotate = -90.0f;
                    }
                    else if (path == "scenes/car.glb") {
                        meshTranslate = glm::vec3(1.0f, 0.0f, -2.0f);
                        meshRotate = 45.0f;
                        meshRotateDir = glm::normalize(glm::vec3(.3, -0.7, 0));
                    }

                    // setup material ids
                    int resolvedMaterialId = 0;
                    if (primitive.material >= 0) {
                        auto existingMaterialEntry = materialIdMap.find(primitive.material);
                        if (existingMaterialEntry != materialIdMap.end()) {
                            resolvedMaterialId = existingMaterialEntry->second;
                        } else {
                            const tinygltf::Material& gltfMaterial = model.materials[primitive.material];
                            const auto& baseColor = gltfMaterial.pbrMetallicRoughness.baseColorFactor;

                            Material newMaterial{};
                            newMaterial.color = glm::vec3(baseColor[0], baseColor[1], baseColor[2]);

                            // uv, textures
                            newMaterial.textureId = -1;
                            int textureIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
                            if (textureIndex >= 0) {
                                int imageIndex = model.textures[textureIndex].source;
                                if (imageIndex >= 0) {
                                    const tinygltf::Image& image = model.images[imageIndex];
                                    if (!image.image.empty()) {
                                        Texture newTexture;
                                        newTexture.offset = (int)textureData.size();
                                        newTexture.width = image.width;
                                        newTexture.height = image.height;
                                        newTexture.channels = image.component;
                                        textureData.insert(textureData.end(), image.image.begin(), image.image.end());
                                        textures.push_back(newTexture);
                                        newMaterial.textureId = (int)textures.size() - 1;
                                    }
                                }
                            }

                            materials.push_back(newMaterial);
                            resolvedMaterialId = (int)materials.size() - 1;
                            materialIdMap[primitive.material] = resolvedMaterialId;
                        }
                    }

                    // setup uv for custom mats
                    bool hasUV = false;
                    const tinygltf::Accessor* uvAccessor = nullptr;
                    const tinygltf::BufferView* uvBufferView = nullptr;
                    const tinygltf::Buffer* uvBuffer = nullptr;
                    size_t uvStride = 0;

                    auto uvAttributeEntry = primitive.attributes.find("TEXCOORD_0");
                    if (uvAttributeEntry != primitive.attributes.end()) {
                        hasUV = true;
                        uvAccessor = &model.accessors[uvAttributeEntry->second];
                        uvBufferView = &model.bufferViews[uvAccessor->bufferView];
                        uvBuffer = &model.buffers[uvBufferView->buffer];
                        uvStride = uvBufferView->byteStride ? uvBufferView->byteStride : 2 * sizeof(float);
                    }

                    // build the triangles
                    auto positionAccessorIndex = primitive.attributes.at("POSITION");
                    const tinygltf::Accessor& positionAccessor = model.accessors[positionAccessorIndex];
                    const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
                    const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];
                    int positionStride = positionBufferView.byteStride ? positionBufferView.byteStride : 3 * sizeof(float);

                    for (int i = 0; i + 2 < indices.size(); i += 3) {
                        int i0 = indices[i];
                        int i1 = indices[i + 1];
                        int i2 = indices[i + 2];

                        int positionByteOffset0 = positionBufferView.byteOffset + positionAccessor.byteOffset + i0 * positionStride;
                        int positionByteOffset1 = positionBufferView.byteOffset + positionAccessor.byteOffset + i1 * positionStride;
                        int positionByteOffset2 = positionBufferView.byteOffset + positionAccessor.byteOffset + i2 * positionStride;

                        const float* positionData0 = reinterpret_cast<const float*>(&positionBuffer.data[positionByteOffset0]);
                        const float* positionData1 = reinterpret_cast<const float*>(&positionBuffer.data[positionByteOffset1]);
                        const float* positionData2 = reinterpret_cast<const float*>(&positionBuffer.data[positionByteOffset2]);

                        Triangle temp;
                        glm::mat4 meshTransform =
                            glm::translate(glm::mat4(1.0f), meshTranslate) *
                            glm::rotate(glm::mat4(1.0f), glm::radians(meshRotate), meshRotateDir) *
                            glm::scale(glm::mat4(1.0f), meshScale);
                        temp.v0 = glm::vec3(meshTransform * glm::vec4(positionData0[0], positionData0[1], positionData0[2], 1.0f));
                        temp.v1 = glm::vec3(meshTransform * glm::vec4(positionData1[0], positionData1[1], positionData1[2], 1.0f));
                        temp.v2 = glm::vec3(meshTransform * glm::vec4(positionData2[0], positionData2[1], positionData2[2], 1.0f));
                        temp.normal = glm::normalize(glm::cross(temp.v1 - temp.v0, temp.v2 - temp.v0));
                        temp.materialid = resolvedMaterialId;

                        // uv logic
                        if (hasUV) {
                            int uvByteOffset0 = uvBufferView->byteOffset + uvAccessor->byteOffset + i0 * uvStride;
                            int uvByteOffset1 = uvBufferView->byteOffset + uvAccessor->byteOffset + i1 * uvStride;
                            int uvByteOffset2 = uvBufferView->byteOffset + uvAccessor->byteOffset + i2 * uvStride;

                            const float* uvData0 = reinterpret_cast<const float*>(&uvBuffer->data[uvByteOffset0]);
                            const float* uvData1 = reinterpret_cast<const float*>(&uvBuffer->data[uvByteOffset1]);
                            const float* uvData2 = reinterpret_cast<const float*>(&uvBuffer->data[uvByteOffset2]);

                            temp.uv0 = glm::vec2(uvData0[0], uvData0[1]);
                            temp.uv1 = glm::vec2(uvData1[0], uvData1[1]);
                            temp.uv2 = glm::vec2(uvData2[0], uvData2[1]);
                        } else {
                            temp.uv0 = temp.uv1 = temp.uv2 = glm::vec2(0.0f);
                        }

                        triangleArray.push_back(temp);
                    }
                }
            }
        }
    }

    buildTree(triangleArray, nodes, 0, triangleArray.size());

    return triangleArray;
}

void Scene::loadFromGltfWrapper(const std::string& path)
{
    // hardcoded light for gltf, no more original scene
    Material lightMat{};
    lightMat.color = glm::vec3(1.0f, 1.0f, 1.0f);
    lightMat.emittance = 5.0f;
    lightMat.textureId = -1;
    materials.push_back(lightMat);

    Geom light;
    if (path == "scenes/shiba/scene.gltf") {
        Material floorMat{};
        floorMat.color = glm::vec3(0.5f, 0.5f, 0.5f);
        floorMat.textureId = -1;
        materials.push_back(floorMat);

        Geom newGeom;
        newGeom.type = CUBE;
        newGeom.materialid = (int)materials.size() - 1;
        newGeom.translation = glm::vec3(0.0f);
        newGeom.rotation = glm::vec3(0.0f);
        newGeom.scale = glm::vec3(30.0f, 0.3f, 30.0f);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        geoms.push_back(newGeom);
    }
    if (path == "scenes/the_billiards_room/scene.gltf") {
        light.type = CUBE;
        light.materialid = 0;
        light.translation = glm::vec3(0.0f, 12.0f, 0.0f);
        light.rotation = glm::vec3(0.0f);
        light.scale = glm::vec3(50.0f, 0.3f, 50.0f);
        light.transform = utilityCore::buildTransformationMatrix(light.translation, light.rotation, light.scale);
        light.inverseTransform = glm::inverse(light.transform);
        light.invTranspose = glm::inverseTranspose(light.transform);
        geoms.push_back(light);
    }
    else {
        light.type = CUBE;
        light.materialid = 0;
        light.rotation = glm::vec3(0.0f);
        light.scale = glm::vec3(50.0f, 0.3f, 50.0f);
        light.translation = glm::vec3(0.0f, 30.0f, 0.0f);
        light.transform = utilityCore::buildTransformationMatrix(light.translation, light.rotation, light.scale);
        light.inverseTransform = glm::inverse(light.transform);
        light.invTranspose = glm::inverseTranspose(light.transform);
        geoms.push_back(light);
    }

    // hardcoded camera
    Camera& camera = state.camera;
    RenderState& stateRef = this->state;
    camera.resolution = glm::ivec2(800, 800);
    float fovy = 45.0f;
    stateRef.iterations = 5000;
    stateRef.traceDepth = 8;
    stateRef.imageName = path.substr(path.find_last_of('/'));
    camera.position = glm::vec3(0.0f, 5.0f, 10.5f);
    camera.lookAt = glm::vec3(0.0f, 5.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);
    camera.view = glm::normalize(camera.lookAt - camera.position);

    int arraylen = camera.resolution.x * camera.resolution.y;
    stateRef.image.resize(arraylen);
    std::fill(stateRef.image.begin(), stateRef.image.end(), glm::vec3());

    // load mesh
    triangles = loadFromGltf(path, materials, nodes, textures, textureData);
    meshBoundsMin = glm::vec3(1e38f);
    meshBoundsMax = glm::vec3(-1e38f);
    for (const auto& temp : triangles)
    {
        meshBoundsMin = glm::min(meshBoundsMin, glm::min(temp.v0, glm::min(temp.v1, temp.v2)));
        meshBoundsMax = glm::max(meshBoundsMax, glm::max(temp.v0, glm::max(temp.v1, temp.v2)));
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.textureId = -1;
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
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
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
