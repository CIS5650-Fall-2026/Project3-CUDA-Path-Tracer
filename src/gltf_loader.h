#pragma once
#ifndef TINYGLTF3_ENABLE_FS
#define TINYGLTF3_ENABLE_FS
#endif

#include <vector>
#include <string>
#include <iostream>
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include "tiny_gltf_v3.h"

#include <filesystem>

// Helper function to extract element indices regardless of type (uint8, uint16, uint32)
inline uint32_t getIndex(const tg3_model& model, const tg3_accessor& accessor, size_t idx) {
    if (accessor.buffer_view < 0 || accessor.buffer_view >= static_cast<int32_t>(model.buffer_views_count)) return 0;
    const tg3_buffer_view& bufferView = model.buffer_views[accessor.buffer_view];
    if (bufferView.buffer < 0 || bufferView.buffer >= static_cast<int32_t>(model.buffers_count)) return 0;
    const tg3_buffer& buffer = model.buffers[bufferView.buffer];

    const uint8_t* data = buffer.data.data + bufferView.byte_offset + accessor.byte_offset;

    switch (accessor.component_type) {
    case TG3_COMPONENT_TYPE_UNSIGNED_INT: {
        const uint32_t* buf = reinterpret_cast<const uint32_t*>(data + idx * sizeof(uint32_t));
        return *buf;
    }
    case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(data + idx * sizeof(uint16_t));
        return static_cast<uint32_t>(*buf);
    }
    case TG3_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const uint8_t* buf = reinterpret_cast<const uint8_t*>(data + idx * sizeof(uint8_t));
        return static_cast<uint32_t>(*buf);
    }
    default:
        return 0;
    }
}

// Helper to look up vertex attribute accessors by string key
inline int findAttributeAccessor(const tg3_primitive& primitive, const char* name) {
    for (uint32_t i = 0; i < primitive.attributes_count; ++i) {
        if (tg3_str_equals_cstr(primitive.attributes[i].key, name)) {
            return primitive.attributes[i].value;
        }
    }
    return -1;
}

inline bool loadGLTF(
    const std::string& filename,
    std::vector<Triangle>& out_triangles,
    Geom& meshGeom)
{
    tg3_model model;
    tg3_parse_options options;
    tg3_parse_options_init(&options);

    tg3_error_stack errors;
    tg3_error_stack_init(&errors);

    // Add inside loadGLTF before tg3_parse_file:
    std::cout << "[DEBUG] Current Working Dir: " << std::filesystem::current_path() << std::endl;
    std::cout << "[DEBUG] Target File Path:    " << filename << std::endl;
    std::cout << "[DEBUG] File Exists Check:   " << (std::filesystem::exists(filename) ? "YES" : "NO") << std::endl;

    int32_t ret = tg3_parse_file(&model, &errors, filename.c_str(), static_cast<uint32_t>(filename.length()), &options);

    if (tg3_errors_has_error(&errors) || ret != 0) {
        for (uint32_t i = 0; i < tg3_errors_count(&errors); ++i) {
            const tg3_error_entry* err = tg3_errors_get(&errors, i);
            if (err) {
                if (err->severity == TG3_SEVERITY_ERROR) {
                    std::cerr << "glTF Error: " << (err->message ? err->message : "") << std::endl;
                }
                else {
                    std::cout << "glTF Warning: " << (err->message ? err->message : "") << std::endl;
                }
            }
        }
        tg3_error_stack_free(&errors);
        return false;
    }

    tg3_error_stack_free(&errors);

    meshGeom.triangleStartIdx = static_cast<int>(out_triangles.size());
    AABB box;

    // Traverse all meshes in the glTF document
    for (uint32_t m = 0; m < model.meshes_count; ++m) {
        const tg3_mesh& mesh = model.meshes[m];
        for (uint32_t p = 0; p < mesh.primitives_count; ++p) {
            const tg3_primitive& primitive = mesh.primitives[p];

            // Only support triangle primitives
            if (primitive.mode != TG3_MODE_TRIANGLES) continue;

            // --- 1. Get Positions ---
            int posAccIdx = findAttributeAccessor(primitive, "POSITION");
            if (posAccIdx < 0 || posAccIdx >= static_cast<int32_t>(model.accessors_count)) continue;

            const tg3_accessor& posAccessor = model.accessors[posAccIdx];
            if (posAccessor.buffer_view < 0 || posAccessor.buffer_view >= static_cast<int32_t>(model.buffer_views_count)) continue;
            const tg3_buffer_view& posView = model.buffer_views[posAccessor.buffer_view];
            if (posView.buffer < 0 || posView.buffer >= static_cast<int32_t>(model.buffers_count)) continue;
            const tg3_buffer& posBuffer = model.buffers[posView.buffer];

            const float* positions = reinterpret_cast<const float*>(
                posBuffer.data.data + posView.byte_offset + posAccessor.byte_offset);
            size_t posStride = posView.byte_stride ? (posView.byte_stride / sizeof(float)) : 3;

            // --- 2. Get Normals (Optional) ---
            const float* normals = nullptr;
            size_t normStride = 3;
            int normAccIdx = findAttributeAccessor(primitive, "NORMAL");
            if (normAccIdx >= 0 && normAccIdx < static_cast<int32_t>(model.accessors_count)) {
                const tg3_accessor& normAccessor = model.accessors[normAccIdx];
                if (normAccessor.buffer_view >= 0 && normAccessor.buffer_view < static_cast<int32_t>(model.buffer_views_count)) {
                    const tg3_buffer_view& normView = model.buffer_views[normAccessor.buffer_view];
                    if (normView.buffer >= 0 && normView.buffer < static_cast<int32_t>(model.buffers_count)) {
                        const tg3_buffer& normBuffer = model.buffers[normView.buffer];
                        normals = reinterpret_cast<const float*>(
                            normBuffer.data.data + normView.byte_offset + normAccessor.byte_offset);
                        normStride = normView.byte_stride ? (normView.byte_stride / sizeof(float)) : 3;
                    }
                }
            }

            // --- 3. Get TexCoords / UVs (Optional) ---
            const float* uvs = nullptr;
            size_t uvStride = 2;
            int uvAccIdx = findAttributeAccessor(primitive, "TEXCOORD_0");
            if (uvAccIdx >= 0 && uvAccIdx < static_cast<int32_t>(model.accessors_count)) {
                const tg3_accessor& uvAccessor = model.accessors[uvAccIdx];
                if (uvAccessor.buffer_view >= 0 && uvAccessor.buffer_view < static_cast<int32_t>(model.buffer_views_count)) {
                    const tg3_buffer_view& uvView = model.buffer_views[uvAccessor.buffer_view];
                    if (uvView.buffer >= 0 && uvView.buffer < static_cast<int32_t>(model.buffers_count)) {
                        const tg3_buffer& uvBuffer = model.buffers[uvView.buffer];
                        uvs = reinterpret_cast<const float*>(
                            uvBuffer.data.data + uvView.byte_offset + uvAccessor.byte_offset);
                        uvStride = uvView.byte_stride ? (uvView.byte_stride / sizeof(float)) : 2;
                    }
                }
            }

            // --- 4. Read Triangles (Indexed vs Non-Indexed) ---
            if (primitive.indices >= 0 && primitive.indices < static_cast<int32_t>(model.accessors_count)) {
                // Indexed Geometry
                const tg3_accessor& indexAccessor = model.accessors[primitive.indices];
                size_t numTriangles = indexAccessor.count / 3;

                for (size_t i = 0; i < numTriangles; i++) {
                    Triangle tri;
                    uint32_t i0 = getIndex(model, indexAccessor, i * 3 + 0);
                    uint32_t i1 = getIndex(model, indexAccessor, i * 3 + 1);
                    uint32_t i2 = getIndex(model, indexAccessor, i * 3 + 2);

                    // Positions
                    tri.p0 = glm::vec3(positions[i0 * posStride + 0], positions[i0 * posStride + 1], positions[i0 * posStride + 2]);
                    tri.p1 = glm::vec3(positions[i1 * posStride + 0], positions[i1 * posStride + 1], positions[i1 * posStride + 2]);
                    tri.p2 = glm::vec3(positions[i2 * posStride + 0], positions[i2 * posStride + 1], positions[i2 * posStride + 2]);

                    // Expand AABBs
                    tri.aabb.grow(tri.p0);
                    tri.aabb.grow(tri.p1);
                    tri.aabb.grow(tri.p2);
                    box.grow(tri.aabb);

                    // Normals
                    if (normals) {
                        tri.n0 = glm::vec3(normals[i0 * normStride + 0], normals[i0 * normStride + 1], normals[i0 * normStride + 2]);
                        tri.n1 = glm::vec3(normals[i1 * normStride + 0], normals[i1 * normStride + 1], normals[i1 * normStride + 2]);
                        tri.n2 = glm::vec3(normals[i2 * normStride + 0], normals[i2 * normStride + 1], normals[i2 * normStride + 2]);
                    }
                    else {
                        glm::vec3 faceNormal = glm::normalize(glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
                        tri.n0 = tri.n1 = tri.n2 = faceNormal;
                    }

                    // UVs
                    if (uvs) {
                        tri.uv0 = glm::vec2(uvs[i0 * uvStride + 0], uvs[i0 * uvStride + 1]);
                        tri.uv1 = glm::vec2(uvs[i1 * uvStride + 0], uvs[i1 * uvStride + 1]);
                        tri.uv2 = glm::vec2(uvs[i2 * uvStride + 0], uvs[i2 * uvStride + 1]);
                    }

                    tri.materialId = meshGeom.materialid;
                    out_triangles.push_back(tri);
                }
            }
            else {
                // Non-indexed Geometry
                size_t numTriangles = posAccessor.count / 3;
                for (size_t i = 0; i < numTriangles; i++) {
                    Triangle tri;
                    size_t i0 = i * 3 + 0;
                    size_t i1 = i * 3 + 1;
                    size_t i2 = i * 3 + 2;

                    tri.p0 = glm::vec3(positions[i0 * posStride + 0], positions[i0 * posStride + 1], positions[i0 * posStride + 2]);
                    tri.p1 = glm::vec3(positions[i1 * posStride + 0], positions[i1 * posStride + 1], positions[i1 * posStride + 2]);
                    tri.p2 = glm::vec3(positions[i2 * posStride + 0], positions[i2 * posStride + 1], positions[i2 * posStride + 2]);

                    // Expand AABBs
                    tri.aabb.grow(tri.p0);
                    tri.aabb.grow(tri.p1);
                    tri.aabb.grow(tri.p2);
                    box.grow(tri.aabb);

                    if (normals) {
                        tri.n0 = glm::vec3(normals[i0 * normStride + 0], normals[i0 * normStride + 1], normals[i0 * normStride + 2]);
                        tri.n1 = glm::vec3(normals[i1 * normStride + 0], normals[i1 * normStride + 1], normals[i1 * normStride + 2]);
                        tri.n2 = glm::vec3(normals[i2 * normStride + 0], normals[i2 * normStride + 1], normals[i2 * normStride + 2]);
                    }
                    else {
                        glm::vec3 faceNormal = glm::normalize(glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
                        tri.n0 = tri.n1 = tri.n2 = faceNormal;
                    }

                    tri.materialId = meshGeom.materialid;
                    out_triangles.push_back(tri);
                }
            }
        }
    }

    meshGeom.triangleCount = static_cast<int>(out_triangles.size()) - meshGeom.triangleStartIdx;
    meshGeom.boundingBox = box;

    return true;
}