#include "mesh.h"
#include "json.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using Bytes = std::vector<unsigned char>;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Bytes readBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open mesh buffer: " + path.string());
    }
    return Bytes(std::istreambuf_iterator<char>(file), {});
}

uint32_t readWord(const Bytes& bytes, size_t offset) {
    require(offset <= bytes.size() && bytes.size() - offset >= 4, "Truncated GLB");
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8)
        | (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

Bytes decodeBase64(const std::string& text) {
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    Bytes result;
    unsigned bits = 0;
    int bitCount = 0;
    for (char c : text) {
        if (c == '=') {
            break;
        }
        size_t value = alphabet.find(c);
        require(value != std::string::npos, "Invalid base64 mesh buffer");
        bits = (bits << 6) | unsigned(value);
        bitCount += 6;
        if (bitCount >= 8) {
            bitCount -= 8;
            result.push_back((bits >> bitCount) & 255);
        }
    }
    return result;
}

std::string decodeUri(const std::string& uri) {
    std::string path;
    auto hex = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < uri.size(); i++) {
        if (uri[i] == '%') {
            require(i + 2 < uri.size(), "Invalid buffer URI escape");
            int a = hex(uri[i + 1]), b = hex(uri[i + 2]);
            require(a >= 0 && b >= 0 && (a != 0 || b != 0), "Invalid buffer URI escape");
            path += char(a * 16 + b);
            i += 2;
        } else {
            path += uri[i];
        }
    }
    require(path.find(':') == std::string::npos, "Mesh buffers must use local relative paths");
    return path;
}

struct Accessor {
    const unsigned char* data;
    size_t count;
    size_t stride;
    int component;
};

Accessor readAccessor(const Json& document, const std::vector<Bytes>& buffers,
    int index, bool positions) {
    const auto& accessor = document.at("accessors").at(index);
    require(!accessor.contains("sparse"), "Sparse glTF accessors are not supported");
    require(!accessor.value("normalized", false), "Normalized position/index accessors are not supported");
    int component = accessor.at("componentType");
    require(accessor.at("type") == (positions ? "VEC3" : "SCALAR"), "Wrong mesh accessor type");
    require(positions ? component == 5126
        : (component == 5121 || component == 5123 || component == 5125), "Unsupported mesh component type");
    size_t elementSize = positions ? 12 : (component == 5121 ? 1 : component == 5123 ? 2 : 4);
    const auto& view = document.at("bufferViews").at(accessor.at("bufferView").get<int>());
    const Bytes& buffer = buffers.at(view.at("buffer").get<size_t>());
    size_t start = view.value("byteOffset", size_t(0));
    size_t length = view.at("byteLength");
    size_t offset = accessor.value("byteOffset", size_t(0));
    size_t count = accessor.at("count");
    size_t stride = view.value("byteStride", elementSize);
    require(start <= buffer.size() && length <= buffer.size() - start, "Buffer view exceeds mesh buffer");
    require(count > 0 && stride >= elementSize && offset <= length && elementSize <= length - offset,
        "Invalid mesh accessor range");
    // Divide before multiplying so an invalid count cannot overflow the range check.
    require(count - 1 <= (length - offset - elementSize) / stride, "Mesh accessor exceeds buffer view");
    return {buffer.data() + start + offset, count, stride, component};
}

glm::vec3 positionAt(const Accessor& accessor, size_t index) {
    require(index < accessor.count, "Triangle index exceeds vertex count");
    float values[3];
    std::memcpy(values, accessor.data + index * accessor.stride, sizeof(values));
    require(std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]),
        "Non-finite mesh position");
    return glm::vec3(values[0], values[1], values[2]);
}

uint32_t indexAt(const Accessor& accessor, size_t index) {
    const unsigned char* data = accessor.data + index * accessor.stride;
    uint32_t value = data[0];
    if (accessor.component != 5121) value |= uint32_t(data[1]) << 8;
    if (accessor.component == 5125) value |= (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
    return value;
}

glm::mat4 nodeTransform(const Json& node) {
    if (node.contains("matrix")) {
        const auto& values = node.at("matrix");
        require(values.size() == 16, "Node matrix needs sixteen values");
        glm::mat4 matrix;
        for (int i = 0; i < 16; i++) matrix[i / 4][i % 4] = values.at(i).get<float>();
        return matrix;
    }
    auto t = node.value("translation", std::vector<float>{0, 0, 0});
    auto r = node.value("rotation", std::vector<float>{0, 0, 0, 1});
    auto s = node.value("scale", std::vector<float>{1, 1, 1});
    require(t.size() == 3 && r.size() == 4 && s.size() == 3, "Invalid node transform");
    glm::quat rotation(r[3], r[0], r[1], r[2]);
    require(glm::length(rotation) > 0, "Invalid node quaternion");
    return glm::translate(glm::mat4(1), glm::vec3(t[0], t[1], t[2]))
        * glm::mat4_cast(glm::normalize(rotation))
        * glm::scale(glm::mat4(1), glm::vec3(s[0], s[1], s[2]));
}
}

void loadGltf(const std::string& filename, Geom& geom, std::vector<Triangle>& triangles) {
    try {
        Bytes file = readBytes(filename);
        Json document;
        Bytes binary;
        if (file.size() >= 4 && readWord(file, 0) == 0x46546c67) {
            require(readWord(file, 4) == 2 && readWord(file, 8) == file.size(), "Invalid GLB header");
            bool foundJson = false;
            for (size_t offset = 12; offset < file.size();) {
                uint32_t length = readWord(file, offset);
                uint32_t type = readWord(file, offset + 4);
                offset += 8;
                require(length <= file.size() - offset, "Truncated GLB chunk");
                if (type == 0x4e4f534a) {
                    require(!foundJson && offset == 20, "Invalid GLB JSON chunk");
                    document = Json::parse(file.begin() + offset, file.begin() + offset + length);
                    foundJson = true;
                } else if (type == 0x004e4942) {
                    binary.assign(file.begin() + offset, file.begin() + offset + length);
                }
                offset += length;
            }
            require(foundJson, "GLB has no JSON chunk");
        } else {
            document = Json::parse(file.begin(), file.end());
        }
        require(document.at("asset").at("version") == "2.0", "Only glTF 2.0 is supported");
        require(!document.contains("extensionsRequired") || document.at("extensionsRequired").empty(),
            "Required glTF extensions are not supported");
        std::vector<Bytes> buffers;
        for (const auto& buffer : document.at("buffers")) {
            Bytes bytes;
            if (buffer.contains("uri")) {
                std::string uri = buffer.at("uri");
                if (uri.rfind("data:", 0) == 0) {
                    size_t marker = uri.find(";base64,");
                    require(marker != std::string::npos, "Only base64 data URIs are supported");
                    bytes = decodeBase64(uri.substr(marker + 8));
                } else {
                    bytes = readBytes(std::filesystem::path(filename).parent_path() / decodeUri(uri));
                }
            } else {
                require(buffers.empty() && !binary.empty(), "Missing glTF buffer URI");
                bytes = binary;
            }
            size_t length = buffer.at("byteLength");
            require(length <= bytes.size(), "Mesh buffer is shorter than byteLength");
            bytes.resize(length);
            buffers.push_back(std::move(bytes));
        }

        geom.triangleStart = int(triangles.size());
        geom.boundsMin = glm::vec3(std::numeric_limits<float>::max());
        geom.boundsMax = -geom.boundsMin;
        const auto& nodes = document.at("nodes");
        std::vector<bool> visiting(nodes.size(), false);
        std::function<void(int, glm::mat4)> visit = [&](int nodeIndex, glm::mat4 parent) {
            const auto& node = nodes.at(nodeIndex);
            require(!visiting.at(nodeIndex), "Cycle in glTF node hierarchy");
            visiting[nodeIndex] = true;
            require(!node.contains("skin"), "Skinned glTF meshes are not supported");
            glm::mat4 transform = parent * nodeTransform(node);
            float determinant = glm::determinant(glm::mat3(transform));
            require(std::isfinite(determinant) && determinant != 0.0f, "Singular or invalid mesh transform");
            if (node.contains("mesh")) {
                const auto& mesh = document.at("meshes").at(node.at("mesh").get<int>());
                for (const auto& primitive : mesh.at("primitives")) {
                    require(primitive.value("mode", 4) == 4, "Only glTF TRIANGLES primitives are supported");
                    require(!primitive.contains("targets"), "Morph targets are not supported");
                    Accessor vertices = readAccessor(document, buffers,
                        primitive.at("attributes").at("POSITION"), true);
                    bool indexed = primitive.contains("indices");
                    Accessor indices{};
                    if (indexed) indices = readAccessor(document, buffers, primitive.at("indices"), false);
                    size_t count = indexed ? indices.count : vertices.count;
                    require(count % 3 == 0, "Triangle index count is not a multiple of three");
                    for (size_t i = 0; i < count; i += 3) {
                        glm::vec3 points[3];
                        for (int corner = 0; corner < 3; corner++) {
                            size_t vertex = indexed ? indexAt(indices, i + corner) : i + corner;
                            points[corner] = glm::vec3(transform * glm::vec4(positionAt(vertices, vertex), 1));
                            for (int axis = 0; axis < 3; axis++)
                                require(std::isfinite(points[corner][axis]), "Invalid transformed mesh position");
                        }
                        glm::vec3 normal = glm::cross(points[1] - points[0], points[2] - points[0]);
                        float length = glm::length(normal);
                        if (length == 0.0f) continue;
                        // A mirrored instance reverses winding, not the outside of the object.
                        normal *= (determinant < 0 ? -1.0f : 1.0f) / length;
                        require(triangles.size() < size_t(std::numeric_limits<int>::max()), "Too many mesh triangles");
                        triangles.push_back({points[0], points[1], points[2], normal});
                        for (glm::vec3 point : points) {
                            geom.boundsMin = glm::min(geom.boundsMin, point);
                            geom.boundsMax = glm::max(geom.boundsMax, point);
                        }
                    }
                }
            }
            if (node.contains("children")) {
                for (int child : node.at("children")) visit(child, transform);
            }
            visiting[nodeIndex] = false;
        };

        if (document.contains("scenes")) {
            const auto& scene = document.at("scenes").at(document.value("scene", 0));
            for (int root : scene.at("nodes")) visit(root, geom.transform);
        } else {
            std::vector<bool> child(nodes.size(), false);
            for (const auto& node : nodes) {
                if (node.contains("children"))
                    for (int index : node.at("children")) child.at(index) = true;
            }
            for (int i = 0; i < nodes.size(); i++)
                if (!child[i]) visit(i, geom.transform);
        }
        geom.triangleCount = int(triangles.size()) - geom.triangleStart;
        require(geom.triangleCount > 0, "glTF scene contains no nondegenerate triangles");
        geom.boundsMin -= glm::vec3(0.0001f);
        geom.boundsMax += glm::vec3(0.0001f);
    } catch (const std::exception& error) {
        throw std::runtime_error("Mesh " + filename + ": " + error.what());
    }
}
