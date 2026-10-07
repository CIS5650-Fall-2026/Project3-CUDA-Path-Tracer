#include "meshLoader.h"
#include "tinygltf_include.h"
#include <cfloat>

bool loadMesh(const std::string& path, std::vector<Triangle>& outTriangles, glm::vec3& bboxMin, glm::vec3& bboxMax) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    bool isGlb = path.substr(path.length() - 4) == ".glb";
    if (isGlb) loader.LoadBinaryFromFile(&model, &err, &warn, path);
    else loader.LoadASCIIFromFile(&model, &err, &warn, path);
    auto readVec3 = [&](int accessorIndex) {
        const tinygltf::Accessor& acc = model.accessors[accessorIndex];
        const tinygltf::BufferView& view = model.bufferViews[acc.bufferView];
        const float* data = reinterpret_cast<const float*>(&model.buffers[view.buffer].data[view.byteOffset + acc.byteOffset]);
        std::vector<glm::vec3> out(acc.count);
        for (size_t i = 0; i < acc.count; i++) {
            out[i] = glm::vec3(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
        }
        return out;
    };

    bboxMin = glm::vec3(FLT_MAX);
    bboxMax = glm::vec3(-FLT_MAX);
    size_t startCount = outTriangles.size();
    for (const tinygltf::Mesh& mesh : model.meshes) {
        for (const tinygltf::Primitive& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES || prim.indices < 0) continue;
            if (!prim.attributes.count("POSITION") || !prim.attributes.count("NORMAL")) continue;
            std::vector<glm::vec3> positions = readVec3(prim.attributes.at("POSITION"));
            std::vector<glm::vec3> normals = readVec3(prim.attributes.at("NORMAL"));
            for (const glm::vec3& pos : positions) {
                bboxMin = glm::min(bboxMin, pos);
                bboxMax = glm::max(bboxMax, pos);
            }
            const tinygltf::Accessor& idxAccessor = model.accessors[prim.indices];
            const tinygltf::BufferView& idxView = model.bufferViews[idxAccessor.bufferView];
            const unsigned char* idxBytes = &model.buffers[idxView.buffer].data[idxView.byteOffset + idxAccessor.byteOffset];
            std::vector<int> indices(idxAccessor.count);
            for (size_t i = 0; i < idxAccessor.count; i++) {
                if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    indices[i] = idxBytes[i];
                }
                else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    indices[i] = reinterpret_cast<const unsigned short*>(idxBytes)[i];
                }
                else {
                    indices[i] = (int)reinterpret_cast<const unsigned int*>(idxBytes)[i];
                }
            }

            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                Triangle tri;
                tri.v0 = positions[indices[i]];
                tri.v1 = positions[indices[i + 1]];
                tri.v2 = positions[indices[i + 2]];
                tri.n0 = normals[indices[i]];
                tri.n1 = normals[indices[i + 1]];
                tri.n2 = normals[indices[i + 2]];
                outTriangles.push_back(tri);
            }
        }
    }
    return outTriangles.size() > startCount;
}
