#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cfloat>

using namespace std;
using json = nlohmann::json;

// tinygltf won't load a file with embedded images unless it has an image loader..
// I don't use textures yet so this one just says ok
static bool skipImage(tinygltf::Image*, const int, std::string*, std::string*,
int, int, const unsigned char*, int, void*) {
    return true;
}

static const unsigned char* accessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t elementSize, int& stride) {
    if (accessor.bufferView < 0 || accessor.sparse.isSparse || accessor.count == 0) {
        return nullptr;
    }

    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    stride = accessor.ByteStride(view);
    const size_t start = view.byteOffset + accessor.byteOffset;
    if (stride <= 0 || start + (accessor.count - 1) * stride + elementSize > buffer.data.size()) {
        return nullptr;
    }
    return buffer.data.data() + start;
}

// reads the i-th vec3 out of the raw bytes
static glm::vec3 readVec3(const unsigned char* data, int stride, size_t i) {
    glm::vec3 v;
    memcpy(&v, data + i * stride, sizeof(glm::vec3));
    return v;
}

// how many bytes one index takes, 0 means an index type i don't support
static size_t componentSize(int componentType) {
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return 1;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: return 4;
        default: return 0;
    }
}

static uint32_t readIndex(const unsigned char* data, int stride, int componentType, size_t i) {
    const unsigned char* p = data + i * stride;
    if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        return *p;
    }
    if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        uint16_t index;
        memcpy(&index, p, sizeof(index));
        return index;
    }
    uint32_t index;
    memcpy(&index, p, sizeof(index));
    return index;
}

// turns one glTF primitive into world-space triangles and adds them to the list
static void appendPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive, const glm::mat4& transform, vector<Triangle>& triangles) {
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
        cout << "Skipping a glTF primitive that is not a triangle list (mode )"
            << primitive.mode << endl;
        return;
    }

    const auto positionAttribute = primitive.attributes.find("POSITION");
    if (positionAttribute == primitive.attributes.end()) {
        return;
    }
    const tinygltf::Accessor& positionAccessor = model.accessors[positionAttribute->second];
    int positionStride = 0;
    const unsigned char* positions = nullptr;
    if (positionAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && positionAccessor.type == TINYGLTF_TYPE_VEC3) {
        positions = accessorData(model, positionAccessor, sizeof(glm::vec3), positionStride);
    }
    if (positions == nullptr) {
        cout << "Skipping a glTF primitive with unsupported vertex positions" << endl;
        return;
    }

    int normalStride = 0;
    const unsigned char* normals = nullptr;
    const auto normalAttribute = primitive.attributes.find("NORMAL");
    if (normalAttribute != primitive.attributes.end()) {
        const tinygltf::Accessor& normalAccessor = model.accessors[normalAttribute->second];
        if (normalAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
            && normalAccessor.type == TINYGLTF_TYPE_VEC3
            && normalAccessor.count == positionAccessor.count) {
                normals = accessorData(model, normalAccessor, sizeof(glm::vec3), normalStride);
            }
    }

    int indexStride = 0;
    int indexType = 0;
    const unsigned char* indices = nullptr;
    size_t vertexCount = positionAccessor.count;
    if (primitive.indices >= 0) {
        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        indexType = indexAccessor.componentType;
        if (indexAccessor.type == TINYGLTF_TYPE_SCALAR && componentSize(indexType) != 0) {
            indices = accessorData(model, indexAccessor, componentSize(indexType), indexStride);
        }
        if (indices == nullptr) {
            cout << "Skipping a glTF primitive with unsupporte indices" << endl;
            return;
        }
        vertexCount = indexAccessor.count;
    }

    // building the triangles
    const glm::mat3 normalTransform = glm::inverseTranspose(glm::mat3(transform));
    const float winding = glm::determinant(glm::mat3(transform)) < 0.0f ? -1.0f : 1.0f;

    for (size_t i = 0; i + 2 < vertexCount; i += 3) {
        uint32_t index[3];
        bool inRange = true;
        for (int k = 0; k < 3; k++) {
            index[k] = indices ? readIndex(indices, indexStride, indexType, i + k) : (uint32_t)(i + k);
            inRange = inRange && index[k] < positionAccessor.count;
        }
        if (!inRange) {
            continue;
        }

        Triangle tri;
        tri.v0 = glm::vec3(transform * glm::vec4(readVec3(positions, positionStride, index[0]), 1.0f));
        tri.v1 = glm::vec3(transform * glm::vec4(readVec3(positions, positionStride, index[1]), 1.0f));
        tri.v2 = glm::vec3(transform * glm::vec4(readVec3(positions, positionStride, index[2]), 1.0f));

        glm::vec3 faceNormal = glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
        if (!(glm::dot(faceNormal, faceNormal) > 0.0f)) {
            continue;
        }
        faceNormal = winding * glm::normalize(faceNormal);

        glm::vec3 n[3];
        for (int k = 0; k < 3; k++) {
            n[k] = normals ? normalTransform * readVec3(normals, normalStride, index[k]) : faceNormal;
            n[k] = glm::dot(n[k], n[k]) > 0.0f ? glm::normalize(n[k]) : faceNormal;
        }
        tri.n0 = n[0];
        tri.n1 = n[1];
        tri.n2 = n[2];

        triangles.push_back(tri);
    }
}

static glm::mat4 nodeTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        return glm::mat4(glm::make_mat4(node.matrix.data()));
    }

    glm::mat4 transform(1.0f);
    if (node.translation.size() == 3) {
        transform = glm::translate(transform, glm::vec3(node.translation[0],
        node.translation[1], node.translation[2]));
    }
    if (node.rotation.size() == 4) {
        const glm::quat rotation(
            (float)node.rotation[3], (float)node.rotation[0],
            (float)node.rotation[1], (float)node.rotation[2]);
        transform *= glm::mat4_cast(rotation);
    }
    if (node.scale.size() == 3) {
        transform = glm::scale(transform,
        glm:: vec3(node.scale[0], node.scale[1], node.scale[2]));
    }
    return transform;
}

static void appendNode(const tinygltf::Model& model, int nodeIndex,
    const glm::mat4& parentTransform, vector<Triangle>& triangles) {
    if (nodeIndex < 0 || nodeIndex >= (int)model.nodes.size()) {
        return;
    }

    const tinygltf::Node& node = model.nodes[nodeIndex];
    const glm::mat4 transform = parentTransform * nodeTransform(node);
    if (node.mesh >= 0 && node.mesh < (int)model.meshes.size()) {
        for (const tinygltf::Primitive& primitive : model.meshes[node.mesh].primitives) {
            appendPrimitive(model, primitive, transform, triangles);
        }
    }
    for (int child : node.children) {
        appendNode(model, child, transform, triangles);
    }
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

void Scene::loadGLTF(const std::string& filename, Geom& mesh)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(skipImage, nullptr);

    std::string err;
    std::string warn;
    const size_t dot = filename.find_last_of('.');
    const bool binary = dot != std::string::npos && filename.substr(dot) == ".glb";
    const bool loaded = binary
        ? loader.LoadBinaryFromFile(&model, &err, &warn, filename)
        : loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        cout << "glTF warning in " << filename << ": " << warn << endl;
    }
    if (!loaded) {
        cout << "Couldn't read glTF from " << filename << ": " << err << endl;
        exit(-1);
    }

    mesh.triangleStart = triangles.size();
    if (model.scenes.empty()) {
        for (const tinygltf::Mesh& gltfMesh : model.meshes) {
            for (const tinygltf::Primitive& primitive : gltfMesh.primitives) {
                appendPrimitive(model, primitive, mesh.transform, triangles);
            }
        }
    } else {
        const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        for (int node : model.scenes[sceneIndex].nodes) {
            appendNode(model, node, mesh.transform, triangles);
        }
    }
    mesh.triangleCount = triangles.size() - mesh.triangleStart;
    mesh.bvhRoot = buildBVH(triangles, mesh.triangleStart, mesh.triangleCount, bvhNodes);
    if (mesh.triangleCount == 0) {
        cout << "Warning: glTF file " << filename << " contains no triangles" << endl;
        return;
    }
    mesh.boundsMin = glm::vec3(FLT_MAX);
    mesh.boundsMax = glm::vec3(-FLT_MAX);
    for (int i = mesh.triangleStart; i < mesh.triangleStart + mesh.triangleCount; i++) {
        const Triangle& tri = triangles[i];
        mesh.boundsMin = glm::min(mesh.boundsMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
        mesh.boundsMax = glm::max(mesh.boundsMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
    }
    cout << "Loaded " << filename << ": " << mesh.triangleCount << "triangles, bounds "
        << glm::to_string(mesh.boundsMin) << " to " << glm::to_string(mesh.boundsMax) << endl;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // we run from renders/ with ../scenes/*.json so the model path needs ../scenes/ in front.
    const size_t lastSlash = jsonName.find_last_of("/\\");
    const std::string sceneDirectory = lastSlash == std::string::npos ? "" : jsonName.substr(0, lastSlash + 1);

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
            newMaterial.hasReflective = 1.0f;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom{};
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "gltf") {
            newGeom.type = MESH;
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

        if (newGeom.type == MESH) {
            loadGLTF(sceneDirectory + p["FILE"].get<std::string>(), newGeom);
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
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", 0.0f);


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
