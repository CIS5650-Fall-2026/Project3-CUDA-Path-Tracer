#include "geometry.cuh"
#include "scene.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static_assert(std::is_trivially_copyable<TextureDescriptor>::value,
              "TextureDescriptor must be safe to copy to a device buffer");
static_assert(std::is_trivially_copyable<SceneView>::value,
              "SceneView must remain a non-owning POD view");

namespace {

class AssertionFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

int gAssertions = 0;

[[noreturn]] void fail(const char* file, int line, const std::string& message) {
    std::ostringstream out;
    out << file << ':' << line << ": " << message;
    throw AssertionFailure(out.str());
}

void check(bool condition, const char* expression, const char* file, int line) {
    ++gAssertions;
    if (!condition) {
        fail(file, line, std::string("check failed: ") + expression);
    }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

void checkNear(float actual, float expected, float tolerance, const char* actualExpr,
               const char* expectedExpr, const char* file, int line) {
    ++gAssertions;
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << std::setprecision(9) << actualExpr << " (" << actual << ") differs from "
            << expectedExpr << " (" << expected << ") by more than " << tolerance;
        fail(file, line, out.str());
    }
}

#define CHECK_NEAR(actual, expected, tolerance) \
    checkNear((actual), (expected), (tolerance), #actual, #expected, __FILE__, __LINE__)

void checkVec2Near(const glm::vec2& actual, const glm::vec2& expected, float tolerance,
                   const char* actualExpr, const char* expectedExpr, const char* file, int line) {
    for (int i = 0; i < 2; ++i) {
        checkNear(actual[i], expected[i], tolerance, actualExpr, expectedExpr, file, line);
    }
}

void checkVec3Near(const glm::vec3& actual, const glm::vec3& expected, float tolerance,
                   const char* actualExpr, const char* expectedExpr, const char* file, int line) {
    for (int i = 0; i < 3; ++i) {
        checkNear(actual[i], expected[i], tolerance, actualExpr, expectedExpr, file, line);
    }
}

#define CHECK_VEC2_NEAR(actual, expected, tolerance) \
    checkVec2Near((actual), (expected), (tolerance), #actual, #expected, __FILE__, __LINE__)
#define CHECK_VEC3_NEAR(actual, expected, tolerance) \
    checkVec3Near((actual), (expected), (tolerance), #actual, #expected, __FILE__, __LINE__)

bool finite(const glm::vec2& value) {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

bool finite(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const fs::path base = fs::temp_directory_path();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = base / ("cis5650_geometry_tests_" + std::to_string(ticks) + "_" +
                            std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path_, error)) {
                return;
            }
        }
        throw std::runtime_error("could not create a unique temporary test directory");
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void writeText(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not create fixture " + path.string());
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) {
        throw std::runtime_error("could not write fixture " + path.string());
    }
}

void writeP6(const fs::path& path, int width, int height,
             const std::vector<unsigned char>& rgb) {
    CHECK(width > 0);
    CHECK(height > 0);
    CHECK(rgb.size() == static_cast<std::size_t>(width * height * 3));
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not create PPM fixture " + path.string());
    }
    output << "P6\n" << width << ' ' << height << "\n255\n";
    output.write(reinterpret_cast<const char*>(rgb.data()),
                 static_cast<std::streamsize>(rgb.size()));
    if (!output) {
        throw std::runtime_error("could not write PPM fixture " + path.string());
    }
}

std::string jsonQuote(const std::string& input) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : input) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c) << std::dec;
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    out << '"';
    return out.str();
}

std::string defaultCamera(const std::string& extra = std::string()) {
    return std::string(
        "{\"RES\":[200,100],\"FOVY\":45.0,\"ITERATIONS\":4,\"DEPTH\":3,"
        "\"FILE\":\"test-output\",\"EYE\":[0,0,10],\"LOOKAT\":[0,0,0],"
        "\"UP\":[0,1,0]") + extra + "}";
}

std::string basicSceneJson(const std::string& materials, const std::string& objects,
                           const std::string& camera = defaultCamera(),
                           const std::string& rootExtra = std::string()) {
    return std::string("{\"Materials\":") + materials + ",\"Objects\":" + objects +
           ",\"Camera\":" + camera + rootExtra + "}";
}

const char* kDiffuseMaterial =
    "{\"mat\":{\"TYPE\":\"Diffuse\",\"RGB\":[1,1,1]}}";

std::string objObject(const std::string& file, const std::string& translation = "[0,0,0]",
                      const std::string& rotation = "[0,0,0]",
                      const std::string& scale = "[1,1,1]") {
    return std::string("[{\"TYPE\":\"obj\",\"FILE\":") + jsonQuote(file) +
           ",\"MATERIAL\":\"mat\",\"TRANS\":" + translation +
           ",\"ROTAT\":" + rotation + ",\"SCALE\":" + scale + "}]";
}

Scene loadObjScene(const TemporaryDirectory& temporary, const std::string& stem,
                   const std::string& obj, const std::string& translation = "[0,0,0]",
                   const std::string& rotation = "[0,0,0]",
                   const std::string& scale = "[1,1,1]") {
    const fs::path objectPath = temporary.path() / (stem + ".obj");
    const fs::path scenePath = temporary.path() / (stem + ".json");
    writeText(objectPath, obj);
    writeText(scenePath,
              basicSceneJson(kDiffuseMaterial,
                             objObject(objectPath.filename().generic_string(), translation,
                                       rotation, scale)));
    return Scene(scenePath.string());
}

template <typename Function>
std::string requireRuntimeError(Function&& function) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        ++gAssertions;
        return error.what();
    } catch (const std::exception& error) {
        fail(__FILE__, __LINE__,
             std::string("expected std::runtime_error, got another exception: ") + error.what());
    } catch (...) {
        fail(__FILE__, __LINE__, "expected std::runtime_error, got a non-standard exception");
    }
    fail(__FILE__, __LINE__, "expected std::runtime_error, but no exception was thrown");
}

bool containsLineDiagnostic(const std::string& message, int line) {
    const std::string number = std::to_string(line);
    return message.find(":" + number) != std::string::npos ||
           message.find("line " + number) != std::string::npos ||
           message.find("line " + number + ":") != std::string::npos;
}

void expectObjError(const std::string& stem, const std::string& obj, int line) {
    TemporaryDirectory temporary;
    const fs::path objectPath = temporary.path() / (stem + ".obj");
    const fs::path scenePath = temporary.path() / (stem + ".json");
    writeText(objectPath, obj);
    writeText(scenePath,
              basicSceneJson(kDiffuseMaterial,
                             objObject(objectPath.filename().generic_string())));
    const std::string message = requireRuntimeError([&] { Scene scene(scenePath.string()); });
    CHECK(message.find(objectPath.filename().string()) != std::string::npos);
    CHECK(containsLineDiagnostic(message, line));
}

glm::mat4 buildTransform(const glm::vec3& translation, const glm::vec3& rotation,
                         const glm::vec3& scale) {
    constexpr float radiansPerDegree = 0.01745329251994329577f;
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation);
    transform = transform * glm::rotate(glm::mat4(1.0f), rotation.x * radiansPerDegree,
                                        glm::vec3(1.0f, 0.0f, 0.0f));
    transform = transform * glm::rotate(glm::mat4(1.0f), rotation.y * radiansPerDegree,
                                        glm::vec3(0.0f, 1.0f, 0.0f));
    transform = transform * glm::rotate(glm::mat4(1.0f), rotation.z * radiansPerDegree,
                                        glm::vec3(0.0f, 0.0f, 1.0f));
    return transform * glm::scale(glm::mat4(1.0f), scale);
}

Geom analyticGeom(GeomType type, int materialId, const glm::vec3& translation,
                  const glm::vec3& rotation = glm::vec3(0.0f),
                  const glm::vec3& scale = glm::vec3(1.0f)) {
    Geom geom{};
    geom.type = type;
    geom.materialid = materialId;
    geom.translation = translation;
    geom.rotation = rotation;
    geom.scale = scale;
    geom.transform = buildTransform(translation, rotation, scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
    return geom;
}

Geom triangleGeom(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
                  int materialId = 0) {
    Geom geom = analyticGeom(TRIANGLE, materialId, glm::vec3(0.0f));
    geom.vertices[0] = a;
    geom.vertices[1] = b;
    geom.vertices[2] = c;
    const glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
    geom.normals[0] = normal;
    geom.normals[1] = normal;
    geom.normals[2] = normal;
    geom.hasNormals = true;
    geom.uvs[0] = glm::vec2(0.0f, 0.0f);
    geom.uvs[1] = glm::vec2(1.0f, 0.0f);
    geom.uvs[2] = glm::vec2(0.0f, 1.0f);
    geom.hasUVs = true;
    return geom;
}

SceneView makeView(const std::vector<Geom>& geoms, const std::vector<Material>& materials,
                   const std::vector<BvhNode>& nodes,
                   const std::vector<int>& primitiveIndices,
                   const std::vector<TextureDescriptor>& textures = {},
                   const std::vector<glm::vec3>& texels = {}) {
    SceneView view{};
    view.geoms = geoms.empty() ? nullptr : geoms.data();
    view.geomCount = static_cast<int>(geoms.size());
    view.materials = materials.empty() ? nullptr : materials.data();
    view.materialCount = static_cast<int>(materials.size());
    view.nodes = nodes.empty() ? nullptr : nodes.data();
    view.nodeCount = static_cast<int>(nodes.size());
    view.primitiveIndices = primitiveIndices.empty() ? nullptr : primitiveIndices.data();
    view.textures = textures.empty() ? nullptr : textures.data();
    view.textureCount = static_cast<int>(textures.size());
    view.texels = texels.empty() ? nullptr : texels.data();
    view.texelCount = static_cast<int>(texels.size());
    return view;
}

void compareHits(const Hit& actual, const Hit& expected, float tolerance = 2.0e-4f) {
    CHECK_NEAR(actual.t, expected.t, tolerance);
    CHECK(actual.primitiveId == expected.primitiveId);
    CHECK(actual.materialId == expected.materialId);
    CHECK_VEC3_NEAR(actual.position, expected.position, tolerance);
    CHECK_VEC3_NEAR(actual.geometricNormal, expected.geometricNormal, tolerance);
    CHECK_VEC3_NEAR(actual.shadingNormal, expected.shadingNormal, tolerance);
    CHECK_VEC3_NEAR(actual.dpdu, expected.dpdu, 5.0e-4f);
    CHECK_VEC3_NEAR(actual.dpdv, expected.dpdv, 5.0e-4f);
    CHECK_VEC2_NEAR(actual.uv, expected.uv, tolerance);
    CHECK(actual.frontFace == expected.frontFace);
}

void compareBvhAndBrute(const SceneView& view, const Ray& ray, float tMin, float tMax,
                        bool expectedFound, int expectedPrimitive = -1) {
    Hit brute{};
    Hit bvh{};
    const bool bruteFound = closestHit(view, ray, tMin, tMax, false, brute);
    const bool bvhFound = closestHit(view, ray, tMin, tMax, true, bvh);
    CHECK(bruteFound == expectedFound);
    CHECK(bvhFound == expectedFound);
    CHECK(anyHit(view, ray, tMin, tMax, false) == expectedFound);
    CHECK(anyHit(view, ray, tMin, tMax, true) == expectedFound);
    if (expectedFound) {
        compareHits(bvh, brute);
        if (expectedPrimitive >= 0) {
            CHECK(brute.primitiveId == expectedPrimitive);
        }
    }
}

struct TreeSummary {
    int maxDepth = 0;
    int primitiveCount = 0;
    std::vector<int> primitives;
};

void inspectNode(const std::vector<BvhNode>& nodes, const std::vector<int>& indices,
                 int nodeIndex, int depth, std::vector<bool>& visited, TreeSummary& summary) {
    CHECK(nodeIndex >= 0);
    CHECK(nodeIndex < static_cast<int>(nodes.size()));
    CHECK(!visited[static_cast<std::size_t>(nodeIndex)]);
    visited[static_cast<std::size_t>(nodeIndex)] = true;
    summary.maxDepth = std::max(summary.maxDepth, depth);
    const BvhNode& node = nodes[static_cast<std::size_t>(nodeIndex)];
    CHECK(finite(node.boundsMin));
    CHECK(finite(node.boundsMax));
    CHECK(node.boundsMin.x <= node.boundsMax.x);
    CHECK(node.boundsMin.y <= node.boundsMax.y);
    CHECK(node.boundsMin.z <= node.boundsMax.z);
    if (node.count > 0) {
        CHECK(node.left == -1);
        CHECK(node.right == -1);
        CHECK(node.first >= 0);
        CHECK(node.first + node.count <= static_cast<int>(indices.size()));
        summary.primitiveCount += node.count;
        for (int i = 0; i < node.count; ++i) {
            summary.primitives.push_back(indices[static_cast<std::size_t>(node.first + i)]);
        }
    } else {
        CHECK(node.count == 0);
        CHECK(node.left >= 0);
        CHECK(node.right >= 0);
        inspectNode(nodes, indices, node.left, depth + 1, visited, summary);
        inspectNode(nodes, indices, node.right, depth + 1, visited, summary);
    }
}

TreeSummary inspectTree(const std::vector<BvhNode>& nodes, const std::vector<int>& indices) {
    CHECK(!nodes.empty());
    std::vector<bool> visited(nodes.size(), false);
    TreeSummary summary;
    inspectNode(nodes, indices, 0, 0, visited, summary);
    CHECK(std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }));
    return summary;
}

void testPodDefaults() {
    Geom geom{};
    CHECK(geom.materialid == -1);
    CHECK_VEC3_NEAR(geom.scale, glm::vec3(1.0f), 0.0f);
    CHECK(!geom.hasNormals);
    CHECK(!geom.hasUVs);

    Material material{};
    CHECK(material.textureId == -1);
    CHECK(material.bumpTextureId == -1);
    CHECK(material.procedural == 0);
    CHECK_NEAR(material.bumpStrength, 1.0f, 0.0f);
    CHECK_NEAR(material.checkerScale, 8.0f, 0.0f);

    Camera camera{};
    CHECK_NEAR(camera.aperture, 0.0f, 0.0f);
    CHECK_NEAR(camera.focusDistance, 1.0f, 0.0f);

    BvhNode node{};
    CHECK(node.left == -1);
    CHECK(node.right == -1);
    CHECK(node.count == 0);

    Hit hit{};
    CHECK(hit.primitiveId == -1);
    CHECK(hit.materialId == -1);

    SceneView view{};
    CHECK(view.geoms == nullptr);
    CHECK(view.geomCount == 0);
    CHECK(view.nodes == nullptr);
    CHECK(view.nodeCount == 0);
    CHECK(view.textureCount == 0);
    CHECK(view.texelCount == 0);
}

void testObjAllIndexFormsAndIndependentNegativeIndices() {
    TemporaryDirectory temporary;
    const std::string obj =
        "# Homogeneous vertices exercise optional w.\n"
        "v 0 0 0 2\n"
        "v 2 0 0 2\n"
        "v 0 2 0 2\n"
        "vt 0.1 0.2 0.5\n"
        "vt 0.3 0.4\n"
        "vt 0.5 0.6\n"
        "vt 0.7 0.8\n"
        "vn 0 0 2\n"
        "vn 0 1 1\n"
        "f 1 2 3\n"
        "f 1/1 2/2 3/3\n"
        "f 1//1 2//2 3//1\n"
        "f -3/-1/-2 -2/-3/-1 -1/-2/-2 # indices resolve per attribute stream\n";
    Scene scene = loadObjScene(temporary, "all-index-forms", obj);
    CHECK(scene.geoms.size() == 4);
    for (const Geom& geom : scene.geoms) {
        CHECK(geom.type == TRIANGLE);
        CHECK(geom.materialid == 0);
        CHECK_VEC3_NEAR(geom.vertices[0], glm::vec3(0.0f, 0.0f, 0.0f), 1.0e-6f);
        CHECK_VEC3_NEAR(geom.vertices[1], glm::vec3(1.0f, 0.0f, 0.0f), 1.0e-6f);
        CHECK_VEC3_NEAR(geom.vertices[2], glm::vec3(0.0f, 1.0f, 0.0f), 1.0e-6f);
    }

    CHECK(!scene.geoms[0].hasUVs);
    CHECK(!scene.geoms[0].hasNormals);
    for (int corner = 0; corner < 3; ++corner) {
        CHECK(finite(scene.geoms[0].uvs[corner]));
        CHECK_VEC3_NEAR(scene.geoms[0].normals[corner], glm::vec3(0.0f, 0.0f, 1.0f),
                        1.0e-6f);
    }

    CHECK(scene.geoms[1].hasUVs);
    CHECK(!scene.geoms[1].hasNormals);
    CHECK_VEC2_NEAR(scene.geoms[1].uvs[0], glm::vec2(0.1f, 0.2f), 1.0e-6f);
    CHECK_VEC2_NEAR(scene.geoms[1].uvs[1], glm::vec2(0.3f, 0.4f), 1.0e-6f);
    CHECK_VEC2_NEAR(scene.geoms[1].uvs[2], glm::vec2(0.5f, 0.6f), 1.0e-6f);

    CHECK(!scene.geoms[2].hasUVs);
    CHECK(scene.geoms[2].hasNormals);
    CHECK_VEC3_NEAR(scene.geoms[2].normals[0], glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(scene.geoms[2].normals[1],
                    glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f)), 1.0e-6f);

    const Geom& negative = scene.geoms[3];
    CHECK(negative.hasUVs);
    CHECK(negative.hasNormals);
    CHECK_VEC2_NEAR(negative.uvs[0], glm::vec2(0.7f, 0.8f), 1.0e-6f);
    CHECK_VEC2_NEAR(negative.uvs[1], glm::vec2(0.3f, 0.4f), 1.0e-6f);
    CHECK_VEC2_NEAR(negative.uvs[2], glm::vec2(0.5f, 0.6f), 1.0e-6f);
    CHECK_VEC3_NEAR(negative.normals[0], glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(negative.normals[1],
                    glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f)), 1.0e-6f);
    CHECK_VEC3_NEAR(negative.normals[2], glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);

    CHECK(!scene.bvhNodes.empty());
    CHECK(scene.bvhPrimitiveIndices.size() == scene.geoms.size());
    const SceneView view = scene.view();
    CHECK(view.geoms == scene.geoms.data());
    CHECK(view.geomCount == static_cast<int>(scene.geoms.size()));
    CHECK(view.nodes == scene.bvhNodes.data());
    CHECK(view.primitiveIndices == scene.bvhPrimitiveIndices.data());
    compareBvhAndBrute(view,
                       Ray{glm::vec3(0.2f, 0.2f, 1.0f), glm::vec3(0.0f, 0.0f, -2.0f)},
                       0.0f, 10.0f, true, 0);

    const fs::path canonicalScene = fs::canonical(temporary.path() / "all-index-forms.json");
    const fs::path canonicalObj = fs::canonical(temporary.path() / "all-index-forms.obj");
    CHECK(std::find(scene.assets.begin(), scene.assets.end(), canonicalScene.string()) !=
          scene.assets.end());
    CHECK(std::find(scene.assets.begin(), scene.assets.end(), canonicalObj.string()) !=
          scene.assets.end());
}

void testObjConvexQuadFanCommentsAndOrganizationalRecords() {
    TemporaryDirectory temporary;
    const std::string obj =
        "# comments and blank lines are legal\n"
        "\n"
        "mtllib ignored.mtl\n"
        "o Plane\n"
        "g Group\n"
        "s 1\n"
        "usemtl ignored_material\n"
        "v -1 -1 0 # lower left\n"
        "v 1 -1 0\n"
        "v 1 1 0\n"
        "v -1 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1 4/4/1 # convex fan\n";
    Scene scene = loadObjScene(temporary, "quad", obj);
    CHECK(scene.geoms.size() == 2);
    const Geom& first = scene.geoms[0];
    const Geom& second = scene.geoms[1];
    CHECK_VEC3_NEAR(first.vertices[0], glm::vec3(-1.0f, -1.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(first.vertices[1], glm::vec3(1.0f, -1.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(first.vertices[2], glm::vec3(1.0f, 1.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(second.vertices[0], glm::vec3(-1.0f, -1.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(second.vertices[1], glm::vec3(1.0f, 1.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(second.vertices[2], glm::vec3(-1.0f, 1.0f, 0.0f), 1.0e-6f);
    CHECK(first.hasUVs && second.hasUVs);
    CHECK(first.hasNormals && second.hasNormals);

    const std::string weaklyConvex =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0.5 1 0\n"
        "v 0 1 0\n"
        "f 1 2 3 4 5\n";
    Scene weak = loadObjScene(temporary, "weakly-convex", weaklyConvex);
    CHECK(weak.geoms.size() == 3);
    for (const Geom& triangle : weak.geoms) {
        CHECK(glm::length(glm::cross(triangle.vertices[1] - triangle.vertices[0],
                                     triangle.vertices[2] - triangle.vertices[0])) > 0.0f);
    }
}

void testObjInvalidGeometryAndSyntax() {
    expectObjError("zero-index", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3\n", 4);
    expectObjError("zero-uv-index",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/0 2/1 3/1\n", 5);
    expectObjError("positive-out-of-range",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n", 4);
    expectObjError("negative-out-of-range",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n", 4);
    expectObjError("duplicate-corner",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 2\n", 4);
    expectObjError("repeated-closing-corner",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 1\n", 4);
    expectObjError("collinear-face",
                   "v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3\n", 4);
    expectObjError("malformed-face-token",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1// 2//1 3//1\n", 5);
    expectObjError("trailing-vertex-junk",
                   "v 0 0 0 trailing\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", 1);
    expectObjError("nonfinite-vertex",
                   "v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", 1);
    expectObjError("zero-homogeneous-w",
                   "v 0 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", 1);
    expectObjError("unsupported-record",
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nl 1 2\nf 1 2 3\n", 4);

    expectObjError("concave-polygon",
                   "v 0 0 0\nv 2 0 0\nv 1 0.5 0\nv 2 2 0\nv 0 2 0\n"
                   "f 1 2 3 4 5\n",
                   6);
    expectObjError("nonplanar-polygon",
                   "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0.1\nf 1 2 3 4\n", 5);
    expectObjError("self-intersecting-polygon",
                   "v 0 0 0\nv 1 1 0\nv 0 1 0\nv 1 0 0\nf 1 2 3 4\n", 5);
}

void testObjSingularTransformAndMissingFile() {
    TemporaryDirectory temporary;
    const std::string obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    const fs::path objectPath = temporary.path() / "triangle.obj";
    const fs::path singularScene = temporary.path() / "singular.json";
    writeText(objectPath, obj);
    writeText(singularScene,
              basicSceneJson(kDiffuseMaterial,
                             objObject(objectPath.filename().generic_string(), "[0,0,0]",
                                       "[0,0,0]", "[1,0,1]")));
    requireRuntimeError([&] { Scene scene(singularScene.string()); });

    const fs::path missingScene = temporary.path() / "missing-obj.json";
    writeText(missingScene,
              basicSceneJson(kDiffuseMaterial, objObject("does-not-exist.obj")));
    const std::string message = requireRuntimeError([&] { Scene scene(missingScene.string()); });
    CHECK(message.find("does-not-exist.obj") != std::string::npos);
}

void testObjMirroredTransformPreservesWindingNormalsAndAttributes() {
    TemporaryDirectory temporary;
    const std::string obj =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1\n";
    Scene scene = loadObjScene(temporary, "mirrored", obj, "[1,2,3]", "[0,90,0]",
                               "[-2,3,4]");
    CHECK(scene.geoms.size() == 1);
    const Geom& triangle = scene.geoms[0];
    const glm::vec3 geometric = glm::normalize(
        glm::cross(triangle.vertices[1] - triangle.vertices[0],
                   triangle.vertices[2] - triangle.vertices[0]));
    const glm::vec3 transformedAuthored = glm::normalize(
        glm::vec3(triangle.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
    CHECK(glm::dot(geometric, transformedAuthored) > 0.999f);
    for (int corner = 0; corner < 3; ++corner) {
        CHECK_VEC3_NEAR(triangle.normals[corner], transformedAuthored, 1.0e-5f);
    }

    const glm::vec3 originalPositions[3] = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
    };
    const glm::vec2 originalUvs[3] = {
        glm::vec2(0.0f, 0.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 1.0f),
    };
    for (int original = 0; original < 3; ++original) {
        const glm::vec3 expectedPosition =
            glm::vec3(triangle.transform * glm::vec4(originalPositions[original], 1.0f));
        int matched = -1;
        for (int corner = 0; corner < 3; ++corner) {
            if (glm::length(triangle.vertices[corner] - expectedPosition) < 1.0e-5f) {
                matched = corner;
            }
        }
        CHECK(matched >= 0);
        CHECK_VEC2_NEAR(triangle.uvs[matched], originalUvs[original], 1.0e-6f);
    }
}

void testSceneMaterialsCameraAndView() {
    TemporaryDirectory temporary;
    const std::string materials =
        "{"
        "\"diff\":{\"TYPE\":\"Diffuse\",\"RGB\":[0.1,0.2,0.3]},"
        "\"emit\":{\"TYPE\":\"Emitting\",\"RGB\":[0.4,0.5,0.6],\"EMITTANCE\":2.5},"
        "\"glass\":{\"TYPE\":\"Glass\",\"RGB\":[0.7,0.8,0.9],\"IOR\":1.5},"
        "\"mirror\":{\"TYPE\":\"Specular\",\"RGB\":[0.9,0.8,0.7]}"
        "}";
    const std::string objects =
        "["
        "{\"TYPE\":\"sphere\",\"MATERIAL\":\"diff\",\"TRANS\":[-3,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]},"
        "{\"TYPE\":\"cube\",\"MATERIAL\":\"emit\",\"TRANS\":[-1,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]},"
        "{\"TYPE\":\"sphere\",\"MATERIAL\":\"glass\",\"TRANS\":[1,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]},"
        "{\"TYPE\":\"cube\",\"MATERIAL\":\"mirror\",\"TRANS\":[3,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}"
        "]";
    const fs::path scenePath = temporary.path() / "materials-camera.json";
    writeText(scenePath, basicSceneJson(materials, objects));
    Scene scene(scenePath.string());
    CHECK(scene.geoms.size() == 4);
    CHECK(scene.materials.size() == 4);

    const Material& diffuse = scene.materials[scene.geoms[0].materialid];
    const Material& emitting = scene.materials[scene.geoms[1].materialid];
    const Material& glass = scene.materials[scene.geoms[2].materialid];
    const Material& mirror = scene.materials[scene.geoms[3].materialid];
    CHECK_VEC3_NEAR(diffuse.color, glm::vec3(0.1f, 0.2f, 0.3f), 1.0e-6f);
    CHECK_NEAR(emitting.emittance, 2.5f, 1.0e-6f);
    CHECK(glass.hasRefractive != 0.0f);
    CHECK_NEAR(glass.indexOfRefraction, 1.5f, 1.0e-6f);
    CHECK(mirror.hasReflective != 0.0f);

    const Camera& camera = scene.state.camera;
    CHECK(camera.resolution == glm::ivec2(200, 100));
    CHECK_VEC3_NEAR(camera.position, glm::vec3(0.0f, 0.0f, 10.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(camera.view, glm::vec3(0.0f, 0.0f, -1.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(camera.right, glm::vec3(1.0f, 0.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(camera.up, glm::vec3(0.0f, 1.0f, 0.0f), 1.0e-6f);
    CHECK_NEAR(glm::dot(camera.view, camera.right), 0.0f, 1.0e-6f);
    CHECK_NEAR(glm::dot(camera.view, camera.up), 0.0f, 1.0e-6f);
    CHECK_NEAR(glm::dot(camera.right, camera.up), 0.0f, 1.0e-6f);
    CHECK_NEAR(camera.fov.y, 45.0f, 1.0e-6f);
    CHECK_NEAR(camera.fov.x, 63.4349488f, 1.0e-4f);
    CHECK_VEC2_NEAR(camera.pixelLength, glm::vec2(0.02f, 0.02f), 1.0e-6f);
    CHECK_NEAR(camera.aperture, 0.0f, 0.0f);
    CHECK_NEAR(camera.focusDistance, 10.0f, 1.0e-6f);
    CHECK(scene.state.iterations == 4u);
    CHECK(scene.state.traceDepth == 3);
    CHECK(scene.state.image.size() == 20000);

    const SceneView view = scene.view();
    CHECK(view.geoms == scene.geoms.data());
    CHECK(view.materials == scene.materials.data());
    CHECK(view.nodes == scene.bvhNodes.data());
    CHECK(view.primitiveIndices == scene.bvhPrimitiveIndices.data());
    CHECK(view.textures == nullptr);
    CHECK(view.textureCount == 0);
    CHECK(view.texels == nullptr);
    CHECK(view.texelCount == 0);

    const fs::path canonicalScene = fs::canonical(scenePath);
    CHECK(std::find(scene.assets.begin(), scene.assets.end(), canonicalScene.string()) !=
          scene.assets.end());
}

void testCameraExplicitLensAndValidationErrors() {
    TemporaryDirectory temporary;
    const std::string object =
        "[{\"TYPE\":\"sphere\",\"MATERIAL\":\"mat\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}]";
    const fs::path lensPath = temporary.path() / "lens.json";
    writeText(lensPath,
              basicSceneJson(kDiffuseMaterial, object,
                             defaultCamera(",\"APERTURE\":0.25,\"FOCUS\":4.5")));
    Scene lens(lensPath.string());
    CHECK_NEAR(lens.state.camera.aperture, 0.25f, 1.0e-6f);
    CHECK_NEAR(lens.state.camera.focusDistance, 4.5f, 1.0e-6f);

    const fs::path zeroFovPath = temporary.path() / "zero-fov-boundary.json";
    const std::string zeroFovCamera =
        "{\"RES\":[10,10],\"FOVY\":0,\"ITERATIONS\":1,\"DEPTH\":1,"
        "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0]}";
    writeText(zeroFovPath, basicSceneJson(kDiffuseMaterial, object, zeroFovCamera));
    Scene zeroFov(zeroFovPath.string());
    CHECK_NEAR(zeroFov.state.camera.fov.y, 0.0f, 0.0f);
    CHECK_VEC2_NEAR(zeroFov.state.camera.pixelLength, glm::vec2(0.0f), 0.0f);

    const std::vector<std::pair<std::string, std::string>> invalidCameras = {
        {"zero-width",
         "{\"RES\":[0,100],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0]}"},
        {"zero-iterations",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":0,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0]}"},
        {"zero-depth",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":0,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0]}"},
        {"wide-fov",
         "{\"RES\":[10,10],\"FOVY\":90,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0]}"},
        {"zero-view",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,1],\"UP\":[0,1,0]}"},
        {"zero-up",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,0,0]}"},
        {"collinear-up",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,0,1]}"},
        {"negative-aperture",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0],"
         "\"APERTURE\":-0.1}"},
        {"zero-focus",
         "{\"RES\":[10,10],\"FOVY\":45,\"ITERATIONS\":1,\"DEPTH\":1,"
         "\"FILE\":\"x\",\"EYE\":[0,0,1],\"LOOKAT\":[0,0,0],\"UP\":[0,1,0],"
         "\"FOCUS\":0}"},
    };
    for (const auto& fixture : invalidCameras) {
        const fs::path scenePath = temporary.path() / (fixture.first + ".json");
        writeText(scenePath, basicSceneJson(kDiffuseMaterial, object, fixture.second));
        requireRuntimeError([&] { Scene scene(scenePath.string()); });
    }
}

void testSceneUnknownAndMalformedInputs() {
    TemporaryDirectory temporary;
    const std::string sphere =
        "[{\"TYPE\":\"sphere\",\"MATERIAL\":\"mat\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}]";

    const fs::path unknownMaterialType = temporary.path() / "unknown-material-type.json";
    writeText(unknownMaterialType,
              basicSceneJson("{\"mat\":{\"TYPE\":\"Metal\",\"RGB\":[1,1,1]}}",
                             sphere));
    requireRuntimeError([&] { Scene scene(unknownMaterialType.string()); });

    const fs::path missingColor = temporary.path() / "missing-color.json";
    writeText(missingColor,
              basicSceneJson("{\"mat\":{\"TYPE\":\"Diffuse\"}}", sphere));
    requireRuntimeError([&] { Scene scene(missingColor.string()); });

    const fs::path negativeEmission = temporary.path() / "negative-emission.json";
    writeText(negativeEmission,
              basicSceneJson(
                  "{\"mat\":{\"TYPE\":\"Emitting\",\"RGB\":[1,1,1],"
                  "\"EMITTANCE\":-1}}",
                  sphere));
    requireRuntimeError([&] { Scene scene(negativeEmission.string()); });

    const fs::path badIor = temporary.path() / "bad-ior.json";
    writeText(badIor,
              basicSceneJson(
                  "{\"mat\":{\"TYPE\":\"Glass\",\"RGB\":[1,1,1],\"IOR\":0}}",
                  sphere));
    requireRuntimeError([&] { Scene scene(badIor.string()); });

    const fs::path unknownReference = temporary.path() / "unknown-reference.json";
    const std::string badReference =
        "[{\"TYPE\":\"sphere\",\"MATERIAL\":\"missing\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}]";
    writeText(unknownReference, basicSceneJson(kDiffuseMaterial, badReference));
    requireRuntimeError([&] { Scene scene(unknownReference.string()); });

    const fs::path unknownGeometry = temporary.path() / "unknown-geometry.json";
    const std::string capsule =
        "[{\"TYPE\":\"capsule\",\"MATERIAL\":\"mat\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}]";
    writeText(unknownGeometry, basicSceneJson(kDiffuseMaterial, capsule));
    requireRuntimeError([&] { Scene scene(unknownGeometry.string()); });

    const fs::path singular = temporary.path() / "singular-analytic.json";
    const std::string singularSphere =
        "[{\"TYPE\":\"sphere\",\"MATERIAL\":\"mat\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,0,1]}]";
    writeText(singular, basicSceneJson(kDiffuseMaterial, singularSphere));
    requireRuntimeError([&] { Scene scene(singular.string()); });

    const fs::path badDepth = temporary.path() / "bad-bvh-depth.json";
    writeText(badDepth, basicSceneJson(kDiffuseMaterial, sphere, defaultCamera(),
                                       ",\"BVH_MAX_DEPTH\":61"));
    requireRuntimeError([&] { Scene scene(badDepth.string()); });

    const fs::path zeroDepth = temporary.path() / "zero-bvh-depth.json";
    writeText(zeroDepth, basicSceneJson(kDiffuseMaterial, sphere, defaultCamera(),
                                        ",\"BVH_MAX_DEPTH\":0"));
    requireRuntimeError([&] { Scene scene(zeroDepth.string()); });

    const fs::path malformedJson = temporary.path() / "malformed.json";
    writeText(malformedJson, "{not valid JSON");
    requireRuntimeError([&] { Scene scene(malformedJson.string()); });

    const fs::path wrongExtension = temporary.path() / "scene.txt";
    writeText(wrongExtension, basicSceneJson(kDiffuseMaterial, sphere));
    requireRuntimeError([&] { Scene scene(wrongExtension.string()); });
    requireRuntimeError([&] { Scene scene((temporary.path() / "absent.json").string()); });
}

void testTextureLoadingColorBumpOrientationAndAssets() {
    TemporaryDirectory temporary;
    const fs::path dataDirectory = temporary.path() / "nested";
    fs::create_directories(dataDirectory);
    const fs::path texturePath = dataDirectory / "pixels.ppm";
    // PPM stores rows top-to-bottom. The bottom-left pixel has nontrivial channels so
    // the manual vertical flip and raw normalized-byte storage are observable.
    writeP6(texturePath, 2, 2,
            {
                255, 0, 0,       0, 255, 0,       // top: red, green
                32, 64, 128,     255, 255, 255,   // bottom: colored, white
            });

    const std::string materials =
        "{\"mat\":{\"TYPE\":\"Diffuse\",\"RGB\":[1,1,1],"
        "\"TEXTURE\":\"pixels.ppm\",\"BUMP\":\"pixels.ppm\","
        "\"BUMP_STRENGTH\":2.5,\"PROCEDURAL\":\"checker\",\"CHECKER_SCALE\":12}}";
    const std::string cube =
        "[{\"TYPE\":\"cube\",\"MATERIAL\":\"mat\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}]";
    const fs::path scenePath = dataDirectory / "textured.json";
    writeText(scenePath, basicSceneJson(materials, cube));
    Scene scene(scenePath.string());
    CHECK(scene.materials.size() == 1);
    const Material& material = scene.materials[0];
    CHECK(material.textureId >= 0);
    CHECK(material.bumpTextureId >= 0);
    CHECK(material.textureId == material.bumpTextureId);
    CHECK(material.procedural != 0);
    CHECK_NEAR(material.bumpStrength, 2.5f, 1.0e-6f);
    CHECK_NEAR(material.checkerScale, 12.0f, 1.0e-6f);
    CHECK(scene.textures.size() == 1);
    CHECK(scene.texels.size() == 4);

    const TextureDescriptor& color = scene.textures[material.textureId];
    const TextureDescriptor& bump = scene.textures[material.bumpTextureId];
    CHECK(color.width == 2 && color.height == 2);
    CHECK(bump.width == 2 && bump.height == 2);
    CHECK(color.offset >= 0 && color.offset + 4 <= static_cast<int>(scene.texels.size()));
    CHECK(bump.offset >= 0 && bump.offset + 4 <= static_cast<int>(scene.texels.size()));
    CHECK(color.offset == bump.offset);

    const glm::vec3 encodedBottomLeft(32.0f / 255.0f, 64.0f / 255.0f,
                                      128.0f / 255.0f);
    CHECK_VEC3_NEAR(scene.texels[static_cast<std::size_t>(color.offset)],
                    encodedBottomLeft, 2.0e-6f);
    CHECK_VEC3_NEAR(scene.texels[static_cast<std::size_t>(bump.offset)], encodedBottomLeft,
                    2.0e-6f);
    CHECK_VEC3_NEAR(scene.texels[static_cast<std::size_t>(color.offset + 1)], glm::vec3(1.0f),
                    1.0e-6f);
    CHECK_VEC3_NEAR(scene.texels[static_cast<std::size_t>(color.offset + 2)],
                    glm::vec3(1.0f, 0.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(scene.texels[static_cast<std::size_t>(color.offset + 3)],
                    glm::vec3(0.0f, 1.0f, 0.0f), 1.0e-6f);

    const fs::path canonicalScene = fs::canonical(scenePath);
    const fs::path canonicalTexture = fs::canonical(texturePath);
    for (const std::string& asset : scene.assets) {
        CHECK(fs::path(asset).is_absolute());
        CHECK(fs::path(asset) == fs::canonical(asset));
    }
    CHECK(std::find(scene.assets.begin(), scene.assets.end(), canonicalScene.string()) !=
          scene.assets.end());
    CHECK(std::find(scene.assets.begin(), scene.assets.end(), canonicalTexture.string()) !=
          scene.assets.end());

    const SceneView view = scene.view();
    CHECK(view.textures == scene.textures.data());
    CHECK(view.textureCount == 1);
    CHECK(view.texels == scene.texels.data());
    CHECK(view.texelCount == 4);
}

void testTextureFailures() {
    TemporaryDirectory temporary;
    const std::string cube =
        "[{\"TYPE\":\"cube\",\"MATERIAL\":\"mat\",\"TRANS\":[0,0,0],"
        "\"ROTAT\":[0,0,0],\"SCALE\":[1,1,1]}]";
    const fs::path missingScene = temporary.path() / "missing-texture.json";
    writeText(missingScene,
              basicSceneJson(
                  "{\"mat\":{\"TYPE\":\"Diffuse\",\"RGB\":[1,1,1],"
                  "\"TEXTURE\":\"missing.ppm\"}}",
                  cube));
    const std::string missingMessage =
        requireRuntimeError([&] { Scene scene(missingScene.string()); });
    CHECK(missingMessage.find("missing.ppm") != std::string::npos);

    const fs::path brokenImage = temporary.path() / "broken.ppm";
    writeText(brokenImage, "this is not an image file\n");
    const fs::path brokenScene = temporary.path() / "broken-texture.json";
    writeText(brokenScene,
              basicSceneJson(
                  "{\"mat\":{\"TYPE\":\"Diffuse\",\"RGB\":[1,1,1],"
                  "\"BUMP\":\"broken.ppm\"}}",
                  cube));
    const std::string brokenMessage =
        requireRuntimeError([&] { Scene scene(brokenScene.string()); });
    CHECK(brokenMessage.find("broken.ppm") != std::string::npos);

    const fs::path validImage = temporary.path() / "valid.ppm";
    writeP6(validImage, 1, 1, {128, 64, 32});
    const fs::path aliasScene = temporary.path() / "strength-alias.json";
    writeText(aliasScene,
              basicSceneJson(
                  "{\"mat\":{\"TYPE\":\"Diffuse\",\"RGB\":[1,1,1],"
                  "\"BUMP\":\"valid.ppm\",\"STRENGTH\":-0.5,"
                  "\"PROCEDURAL\":\"none\"}}",
                  cube));
    Scene alias(aliasScene.string());
    CHECK_NEAR(alias.materials[0].bumpStrength, -0.5f, 0.0f);
    CHECK(alias.materials[0].procedural == 0);

    const fs::path conflictScene = temporary.path() / "strength-conflict.json";
    writeText(conflictScene,
              basicSceneJson(
                  "{\"mat\":{\"TYPE\":\"Diffuse\",\"RGB\":[1,1,1],"
                  "\"BUMP\":\"valid.ppm\",\"BUMP_STRENGTH\":0.25,"
                  "\"STRENGTH\":0.5}}",
                  cube));
    requireRuntimeError([&] { Scene scene(conflictScene.string()); });
}

void testPrimitiveIntersectionsAndBvhAgreement() {
    std::vector<Geom> geoms;
    geoms.push_back(analyticGeom(SPHERE, 0, glm::vec3(-2.0f, 0.0f, 0.0f)));
    geoms.push_back(analyticGeom(CUBE, 1, glm::vec3(2.0f, 0.0f, 0.0f)));
    geoms.push_back(triangleGeom(glm::vec3(-0.5f, -0.5f, 0.0f),
                                 glm::vec3(0.5f, -0.5f, 0.0f),
                                 glm::vec3(0.0f, 0.5f, 0.0f), 2));
    std::vector<Material> materials(3);
    std::vector<BvhNode> nodes;
    std::vector<int> indices;
    buildBvh(geoms, nodes, indices);
    const SceneView view = makeView(geoms, materials, nodes, indices);

    compareBvhAndBrute(view,
                       Ray{glm::vec3(-2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 2.0f)},
                       0.0f, 100.0f, true, 0);
    Hit sphereHit{};
    CHECK(closestHit(view,
                     Ray{glm::vec3(-2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 2.0f)},
                     0.0f, 100.0f, false, sphereHit));
    CHECK_NEAR(sphereHit.t, 1.25f, 1.0e-5f);
    CHECK_VEC3_NEAR(sphereHit.position, glm::vec3(-2.0f, 0.0f, -0.5f), 1.0e-5f);
    CHECK_VEC3_NEAR(sphereHit.geometricNormal, glm::vec3(0.0f, 0.0f, -1.0f), 1.0e-5f);
    CHECK(sphereHit.frontFace);
    CHECK(finite(sphereHit.uv));
    CHECK(finite(sphereHit.dpdu) && finite(sphereHit.dpdv));

    compareBvhAndBrute(view,
                       Ray{glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f)},
                       0.0f, 100.0f, true, 1);
    compareBvhAndBrute(view,
                       Ray{glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -2.0f)},
                       0.0f, 100.0f, true, 2);
    compareBvhAndBrute(view,
                       Ray{glm::vec3(10.0f, 10.0f, 10.0f), glm::vec3(1.0f, 0.0f, 0.0f)},
                       0.0f, 100.0f, false);

    Hit insideSphere{};
    const Ray insideSphereRay{glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};
    CHECK(closestHit(view, insideSphereRay, 0.0f, 100.0f, true, insideSphere));
    CHECK_NEAR(insideSphere.t, 0.5f, 1.0e-5f);
    CHECK(!insideSphere.frontFace);
    CHECK_VEC3_NEAR(insideSphere.geometricNormal, glm::vec3(1.0f, 0.0f, 0.0f), 1.0e-5f);
    CHECK(glm::dot(insideSphere.geometricNormal, insideSphereRay.direction) > 0.0f);
    CHECK(glm::dot(insideSphere.geometricNormal, insideSphere.shadingNormal) >= 0.0f);

    Hit insideCube{};
    const Ray insideCubeRay{glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)};
    CHECK(closestHit(view, insideCubeRay, 0.0f, 100.0f, true, insideCube));
    CHECK_NEAR(insideCube.t, 0.5f, 1.0e-5f);
    CHECK(!insideCube.frontFace);
    CHECK_VEC3_NEAR(insideCube.geometricNormal, glm::vec3(0.0f, 1.0f, 0.0f), 1.0e-5f);

    compareBvhAndBrute(view,
                       Ray{glm::vec3(2.0f, -2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)},
                       0.0f, 100.0f, true, 1);
    compareBvhAndBrute(view,
                       Ray{glm::vec3(2.75f, -2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)},
                       0.0f, 100.0f, false);
    compareBvhAndBrute(view,
                       Ray{glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(1.0f, 0.0f, 0.0f)},
                       0.0f, 100.0f, false);

    CHECK(!anyHit(view,
                  Ray{glm::vec3(-2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 2.0f)},
                  0.0f, 1.0f, true));

    // Keep enough precision in the quadratic to distinguish the radius from a
    // large origin distance. Float-only coefficient arithmetic collapses this
    // case to the sphere center.
    const Ray distantSphereRay{glm::vec3(-2.0f, 0.0f, 10000.0f),
                               glm::vec3(0.0f, 0.0f, -1.0f)};
    Hit distantBrute{};
    Hit distantBvh{};
    CHECK(closestHit(view, distantSphereRay, 0.0f, 20000.0f, false, distantBrute));
    CHECK(closestHit(view, distantSphereRay, 0.0f, 20000.0f, true, distantBvh));
    CHECK(distantBrute.primitiveId == 0);
    CHECK(distantBvh.primitiveId == 0);
    CHECK_NEAR(distantBrute.t, 9999.5f, 1.0e-3f);
    CHECK_NEAR(distantBvh.t, distantBrute.t, 0.0f);

    // "Not normalized" includes finite direction magnitudes whose squared
    // length would underflow or overflow in float arithmetic.
    const Ray tinyDirectionRay{glm::vec3(-2.0f, 0.0f, 2.0f),
                               glm::vec3(0.0f, 0.0f, -1.0e-30f)};
    Hit tinyDirectionHit{};
    CHECK(closestHit(view, tinyDirectionRay, 0.0f, FLT_MAX, true, tinyDirectionHit));
    CHECK(tinyDirectionHit.primitiveId == 0);
    CHECK_NEAR(tinyDirectionHit.t / 1.5e30f, 1.0f, 1.0e-5f);

    const Ray hugeDirectionRay{glm::vec3(2.0f, 0.0f, 2.0f),
                               glm::vec3(0.0f, 0.0f, -1.0e20f)};
    Hit hugeDirectionHit{};
    CHECK(closestHit(view, hugeDirectionRay, 0.0f, 1.0f, true, hugeDirectionHit));
    CHECK(hugeDirectionHit.primitiveId == 1);
    CHECK_NEAR(hugeDirectionHit.t / 1.5e-20f, 1.0f, 1.0e-5f);

    const Ray nearlyParallelCubeRay{glm::vec3(2.0f, 0.0f, -2.0f),
                                   glm::vec3(1.0e-39f, 0.0f, 1.0f)};
    compareBvhAndBrute(view, nearlyParallelCubeRay, 0.0f, 10.0f, true, 1);
    Hit nearlyParallelCubeHit{};
    CHECK(closestHit(view, nearlyParallelCubeRay, 0.0f, 10.0f, true,
                     nearlyParallelCubeHit));
    CHECK_NEAR(nearlyParallelCubeHit.t, 1.5f, 0.0f);

    const Ray tinyTriangleRay{glm::vec3(0.0f, 0.0f, 2.0f),
                              glm::vec3(0.0f, 0.0f, -1.0e-30f)};
    Hit tinyTriangleHit{};
    CHECK(closestHit(view, tinyTriangleRay, 0.0f, FLT_MAX, true, tinyTriangleHit));
    CHECK(tinyTriangleHit.primitiveId == 2);
    CHECK_NEAR(tinyTriangleHit.t / 2.0e30f, 1.0f, 1.0e-5f);
}

void testTriangleBarycentricsNormalsAndDerivatives() {
    std::vector<Geom> geoms{
        triangleGeom(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(2.0f, 0.0f, 0.0f),
                     glm::vec3(0.0f, 3.0f, 0.0f), 7),
    };
    geoms[0].normals[0] = glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f));
    geoms[0].normals[1] = glm::normalize(glm::vec3(0.2f, 0.0f, 1.0f));
    geoms[0].normals[2] = glm::normalize(glm::vec3(0.0f, 0.2f, 1.0f));
    std::vector<Material> materials(8);
    std::vector<BvhNode> nodes;
    std::vector<int> indices;
    buildBvh(geoms, nodes, indices);
    const SceneView view = makeView(geoms, materials, nodes, indices);
    const Ray ray{glm::vec3(0.5f, 0.75f, 2.0f), glm::vec3(0.0f, 0.0f, -2.0f)};
    Hit hit{};
    CHECK(closestHit(view, ray, 0.0f, 10.0f, true, hit));
    CHECK_NEAR(hit.t, 1.0f, 1.0e-6f);
    CHECK(hit.primitiveId == 0);
    CHECK(hit.materialId == 7);
    CHECK_VEC3_NEAR(hit.position, glm::vec3(0.5f, 0.75f, 0.0f), 1.0e-6f);
    CHECK_VEC2_NEAR(hit.uv, glm::vec2(0.25f, 0.25f), 1.0e-6f);
    CHECK_VEC3_NEAR(hit.geometricNormal, glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);
    CHECK(glm::dot(hit.geometricNormal, hit.shadingNormal) > 0.0f);
    CHECK_VEC3_NEAR(hit.dpdu, glm::vec3(2.0f, 0.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3_NEAR(hit.dpdv, glm::vec3(0.0f, 3.0f, 0.0f), 1.0e-6f);
    CHECK(hit.frontFace);

    Hit back{};
    const Ray backRay{glm::vec3(0.5f, 0.75f, -2.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
    CHECK(closestHit(view, backRay, 0.0f, 10.0f, true, back));
    CHECK(!back.frontFace);
    CHECK_VEC3_NEAR(back.geometricNormal, glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);
    CHECK(glm::dot(back.geometricNormal, back.shadingNormal) >= 0.0f);

    geoms[0].hasUVs = false;
    geoms[0].uvs[0] = glm::vec2(0.0f);
    geoms[0].uvs[1] = glm::vec2(0.0f);
    geoms[0].uvs[2] = glm::vec2(0.0f);
    buildBvh(geoms, nodes, indices);
    const SceneView fallbackView = makeView(geoms, materials, nodes, indices);
    Hit fallback{};
    CHECK(closestHit(fallbackView, ray, 0.0f, 10.0f, true, fallback));
    CHECK(finite(fallback.uv));
    CHECK(finite(fallback.dpdu));
    CHECK(finite(fallback.dpdv));
    CHECK(glm::length(fallback.dpdu) > 0.0f);
    CHECK(glm::length(fallback.dpdv) > 0.0f);
}

void testStableClosestTieBreak() {
    std::vector<Geom> geoms{
        analyticGeom(SPHERE, 4, glm::vec3(0.0f)),
        analyticGeom(SPHERE, 9, glm::vec3(0.0f)),
    };
    std::vector<Material> materials(10);
    std::vector<BvhNode> nodes;
    std::vector<int> indices;
    buildBvh(geoms, nodes, indices);
    const SceneView view = makeView(geoms, materials, nodes, indices);
    const Ray ray{glm::vec3(0.0f, 0.0f, -2.0f), glm::vec3(0.0f, 0.0f, 3.0f)};
    for (int iteration = 0; iteration < 4; ++iteration) {
        Hit brute{};
        Hit accelerated{};
        CHECK(closestHit(view, ray, 0.0f, 100.0f, false, brute));
        CHECK(closestHit(view, ray, 0.0f, 100.0f, true, accelerated));
        CHECK(brute.primitiveId == 0);
        CHECK(accelerated.primitiveId == 0);
        CHECK(brute.materialId == 4);
        CHECK(accelerated.materialId == 4);
        CHECK_NEAR(brute.t, 0.5f, 1.0e-6f);
    }
}

void testBvhConstructionDeterminismMedianAndDepthCap() {
    std::vector<Geom> ordered;
    for (int i = 0; i < 8; ++i) {
        const float x = static_cast<float>(i * 2);
        ordered.push_back(triangleGeom(glm::vec3(x, 0.0f, 0.0f),
                                       glm::vec3(x + 0.5f, 0.0f, 0.0f),
                                       glm::vec3(x, 0.5f, 0.0f), i));
    }
    std::vector<BvhNode> firstNodes;
    std::vector<int> firstIndices;
    std::vector<BvhNode> secondNodes;
    std::vector<int> secondIndices;
    buildBvh(ordered, firstNodes, firstIndices);
    buildBvh(ordered, secondNodes, secondIndices);
    CHECK(firstIndices == secondIndices);
    CHECK(firstNodes.size() == secondNodes.size());
    for (std::size_t i = 0; i < firstNodes.size(); ++i) {
        CHECK_VEC3_NEAR(firstNodes[i].boundsMin, secondNodes[i].boundsMin, 0.0f);
        CHECK_VEC3_NEAR(firstNodes[i].boundsMax, secondNodes[i].boundsMax, 0.0f);
        CHECK(firstNodes[i].left == secondNodes[i].left);
        CHECK(firstNodes[i].right == secondNodes[i].right);
        CHECK(firstNodes[i].first == secondNodes[i].first);
        CHECK(firstNodes[i].count == secondNodes[i].count);
    }
    TreeSummary orderedSummary = inspectTree(firstNodes, firstIndices);
    CHECK(orderedSummary.primitiveCount == 8);
    std::sort(orderedSummary.primitives.begin(), orderedSummary.primitives.end());
    for (int i = 0; i < 8; ++i) {
        CHECK(orderedSummary.primitives[static_cast<std::size_t>(i)] == i);
    }
    CHECK(firstNodes[0].count == 0);
    const BvhNode& left = firstNodes[static_cast<std::size_t>(firstNodes[0].left)];
    const BvhNode& right = firstNodes[static_cast<std::size_t>(firstNodes[0].right)];
    CHECK(left.boundsMax.x <= right.boundsMin.x);

    std::vector<Geom> many;
    for (int i = 0; i < 20; ++i) {
        many.push_back(analyticGeom(SPHERE, i, glm::vec3(static_cast<float>(i), 0.0f, 0.0f)));
    }
    std::vector<BvhNode> cappedNodes;
    std::vector<int> cappedIndices;
    buildBvh(many, cappedNodes, cappedIndices, 1);
    TreeSummary capped = inspectTree(cappedNodes, cappedIndices);
    CHECK(capped.primitiveCount == 20);
    CHECK(capped.maxDepth <= 1);
    bool foundLargeLeaf = false;
    for (const BvhNode& candidate : cappedNodes) {
        foundLargeLeaf = foundLargeLeaf || candidate.count > 4;
    }
    CHECK(foundLargeLeaf);

    for (int invalidDepth : {0, 61}) {
        std::vector<BvhNode> depthNodes;
        std::vector<int> depthIndices;
        bool rejected = false;
        try {
            buildBvh(many, depthNodes, depthIndices, invalidDepth);
        } catch (const std::invalid_argument&) {
            rejected = true;
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        ++gAssertions;
        if (!rejected) {
            const TreeSummary clamped = inspectTree(depthNodes, depthIndices);
            CHECK(clamped.primitiveCount == 20);
            CHECK(clamped.maxDepth <= 60);
        }
    }

    const Geom ellipsoid = analyticGeom(SPHERE, 0, glm::vec3(7.0f, -3.0f, 2.0f),
                                         glm::vec3(27.0f, 41.0f, 13.0f),
                                         glm::vec3(0.3f, 4.0f, 1.7f));
    std::vector<BvhNode> ellipsoidNodes;
    std::vector<int> ellipsoidIndices;
    buildBvh(std::vector<Geom>{ellipsoid}, ellipsoidNodes, ellipsoidIndices);
    CHECK(ellipsoidNodes.size() == 1);
    for (int latitude = 0; latitude <= 16; ++latitude) {
        const float phi = -1.57079632679f + 3.14159265359f * latitude / 16.0f;
        for (int longitude = 0; longitude < 32; ++longitude) {
            const float theta = 6.28318530718f * longitude / 32.0f;
            const glm::vec3 local(0.5f * std::cos(phi) * std::cos(theta),
                                  0.5f * std::sin(phi),
                                  0.5f * std::cos(phi) * std::sin(theta));
            const glm::vec3 world(ellipsoid.transform * glm::vec4(local, 1.0f));
            for (int axis = 0; axis < 3; ++axis) {
                CHECK(world[axis] >= ellipsoidNodes[0].boundsMin[axis]);
                CHECK(world[axis] <= ellipsoidNodes[0].boundsMax[axis]);
            }
        }
    }
}

void testEmptyGeometryAndInvalidBvhFallback() {
    std::vector<Geom> emptyGeoms;
    std::vector<BvhNode> emptyNodes(1);
    std::vector<int> emptyIndices{123};
    buildBvh(emptyGeoms, emptyNodes, emptyIndices);
    CHECK(emptyNodes.empty());
    CHECK(emptyIndices.empty());
    const SceneView empty{};
    const Ray ray{glm::vec3(0.0f, 0.0f, -2.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
    Hit emptyHit{};
    CHECK(!closestHit(empty, ray, 0.0f, 100.0f, false, emptyHit));
    CHECK(!closestHit(empty, ray, 0.0f, 100.0f, true, emptyHit));
    CHECK(!anyHit(empty, ray, 0.0f, 100.0f, false));
    CHECK(!anyHit(empty, ray, 0.0f, 100.0f, true));

    std::vector<Geom> geoms{analyticGeom(SPHERE, 0, glm::vec3(0.0f))};
    std::vector<Material> materials(1);
    std::vector<int> indices{0};
    BvhNode invalidRoot{};
    invalidRoot.boundsMin = glm::vec3(-0.5f);
    invalidRoot.boundsMax = glm::vec3(0.5f);
    invalidRoot.left = 17;
    invalidRoot.right = -1;
    invalidRoot.first = 0;
    invalidRoot.count = 0;
    std::vector<BvhNode> invalidNodes{invalidRoot};
    SceneView invalidView = makeView(geoms, materials, invalidNodes, indices);
    Hit invalidHit{};
    CHECK(closestHit(invalidView, ray, 0.0f, 100.0f, true, invalidHit));
    CHECK(invalidHit.primitiveId == 0);
    CHECK(anyHit(invalidView, ray, 0.0f, 100.0f, true));

    // Structural validation must happen before bounds culling. This malformed
    // internal root has bounds that miss the ray; treating the cull as valid
    // would silently discard the brute-force hit.
    invalidNodes[0].boundsMin = glm::vec3(10.0f);
    invalidNodes[0].boundsMax = glm::vec3(11.0f);
    invalidNodes[0].left = -1;
    invalidNodes[0].right = -1;
    invalidNodes[0].first = 0;
    invalidNodes[0].count = 0;
    indices[0] = 0;
    invalidView = makeView(geoms, materials, invalidNodes, indices);
    CHECK(closestHit(invalidView, ray, 0.0f, 100.0f, true, invalidHit));
    CHECK(invalidHit.primitiveId == 0);
    CHECK(anyHit(invalidView, ray, 0.0f, 100.0f, true));

    invalidNodes[0].left = -1;
    invalidNodes[0].right = -1;
    invalidNodes[0].first = 0;
    invalidNodes[0].count = 1;
    indices[0] = 99;
    invalidView = makeView(geoms, materials, invalidNodes, indices);
    CHECK(closestHit(invalidView, ray, 0.0f, 100.0f, true, invalidHit));
    CHECK(invalidHit.primitiveId == 0);
    CHECK(anyHit(invalidView, ray, 0.0f, 100.0f, true));

    // With BVH requested but no tree data, correctness still requires brute force.
    const std::vector<BvhNode> noNodes;
    const std::vector<int> noIndices;
    const SceneView absentTree = makeView(geoms, materials, noNodes, noIndices);
    CHECK(closestHit(absentTree, ray, 0.0f, 100.0f, true, invalidHit));
    CHECK(invalidHit.primitiveId == 0);
    CHECK(anyHit(absentTree, ray, 0.0f, 100.0f, true));
}

void testBvhDepthOrStackOverflowFallsBack() {
    std::vector<Geom> geoms{analyticGeom(SPHERE, 0, glm::vec3(0.0f))};
    std::vector<Material> materials(1);
    constexpr int internalCount = 66;
    std::vector<BvhNode> nodes(static_cast<std::size_t>(internalCount * 2 + 1));
    std::vector<int> indices(static_cast<std::size_t>(internalCount + 1), 0);
    const int leafBase = internalCount;
    for (int i = 0; i < internalCount; ++i) {
        BvhNode& internal = nodes[static_cast<std::size_t>(i)];
        internal.boundsMin = glm::vec3(-1.0f);
        internal.boundsMax = glm::vec3(1.0f);
        internal.left = (i + 1 < internalCount) ? i + 1 : leafBase + internalCount;
        internal.right = leafBase + i;
        internal.first = 0;
        internal.count = 0;
    }
    for (int i = 0; i <= internalCount; ++i) {
        BvhNode& leaf = nodes[static_cast<std::size_t>(leafBase + i)];
        leaf.boundsMin = glm::vec3(-1.0f);
        leaf.boundsMax = glm::vec3(1.0f);
        leaf.left = -1;
        leaf.right = -1;
        leaf.first = i;
        leaf.count = 1;
    }
    const SceneView view = makeView(geoms, materials, nodes, indices);
    const Ray ray{glm::vec3(0.0f, 0.0f, -2.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
    Hit hit{};
    CHECK(closestHit(view, ray, 0.0f, 100.0f, true, hit));
    CHECK(hit.primitiveId == 0);
    CHECK_NEAR(hit.t, 1.5f, 1.0e-6f);
    CHECK(anyHit(view, ray, 0.0f, 100.0f, true));
}

std::uint32_t advanceRandom(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float signedRandom(std::uint32_t& state) {
    const float unit = static_cast<float>(advanceRandom(state) >> 8) *
                       (1.0f / 16777216.0f);
    return 2.0f * unit - 1.0f;
}

void checkHitsExactlyEqual(const Hit& actual, const Hit& expected) {
    CHECK(actual.t == expected.t);
    CHECK(actual.primitiveId == expected.primitiveId);
    CHECK(actual.materialId == expected.materialId);
    for (int axis = 0; axis < 3; ++axis) {
        CHECK(actual.position[axis] == expected.position[axis]);
        CHECK(actual.geometricNormal[axis] == expected.geometricNormal[axis]);
        CHECK(actual.shadingNormal[axis] == expected.shadingNormal[axis]);
        CHECK(actual.dpdu[axis] == expected.dpdu[axis]);
        CHECK(actual.dpdv[axis] == expected.dpdv[axis]);
    }
    CHECK(actual.uv.x == expected.uv.x);
    CHECK(actual.uv.y == expected.uv.y);
    CHECK(actual.frontFace == expected.frontFace);
}

void testRandomizedTransformedBvhEquivalence() {
    std::vector<Geom> geoms;
    // A shared-edge floor, a duplicate coplanar cap, and a cube bottom occupy
    // the same plane. These exercise traversal-order-independent float ties.
    geoms.push_back(triangleGeom(glm::vec3(-3.0f, -1.0f, -3.0f),
                                 glm::vec3(3.0f, -1.0f, 3.0f),
                                 glm::vec3(3.0f, -1.0f, -3.0f), 0));
    geoms.push_back(triangleGeom(glm::vec3(-3.0f, -1.0f, -3.0f),
                                 glm::vec3(-3.0f, -1.0f, 3.0f),
                                 glm::vec3(3.0f, -1.0f, 3.0f), 1));
    geoms.push_back(geoms[0]);
    geoms.back().materialid = 2;
    geoms.push_back(analyticGeom(CUBE, 3, glm::vec3(0.0f, -0.5f, 0.0f)));

    for (int i = 0; i < 16; ++i) {
        const GeomType type = (i % 2 == 0) ? SPHERE : CUBE;
        const glm::vec3 translation(static_cast<float>((i % 4) * 3 - 5),
                                    static_cast<float>((i / 4) * 2),
                                    static_cast<float>((i * 7) % 9 - 4));
        const glm::vec3 rotation(13.0f * i, 29.0f * i, 7.0f * i);
        glm::vec3 scale(0.55f + 0.11f * static_cast<float>(i % 5),
                        0.7f + 0.09f * static_cast<float>((i + 2) % 7),
                        0.6f + 0.13f * static_cast<float>((i + 1) % 4));
        if (i % 5 == 0) {
            scale.x = -scale.x;
        }
        geoms.push_back(analyticGeom(type, 4 + i, translation, rotation, scale));
    }

    geoms.push_back(triangleGeom(glm::vec3(-4.0f, 0.0f, 2.0f),
                                 glm::vec3(-1.0f, 3.0f, 2.5f),
                                 glm::vec3(-2.0f, -0.5f, 3.0f), 20));
    geoms.push_back(triangleGeom(glm::vec3(2.0f, -0.5f, -3.0f),
                                 glm::vec3(5.0f, 1.0f, -2.0f),
                                 glm::vec3(3.0f, 4.0f, -1.0f), 21));

    std::vector<Material> materials(22);
    std::vector<BvhNode> nodes;
    std::vector<int> indices;
    buildBvh(geoms, nodes, indices);
    const SceneView view = makeView(geoms, materials, nodes, indices);

    const Ray contactRay{glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(0.0f, 3.0f, 0.0f)};
    Hit contactBrute{};
    Hit contactBvh{};
    CHECK(closestHit(view, contactRay, 0.0f, 10.0f, false, contactBrute));
    CHECK(closestHit(view, contactRay, 0.0f, 10.0f, true, contactBvh));
    checkHitsExactlyEqual(contactBvh, contactBrute);
    CHECK(contactBrute.primitiveId == 0);

    const float edgeCoordinates[] = {
        -3.0f,
        std::nextafter(-3.0f, 0.0f),
        std::nextafter(-3.0f, -4.0f),
        0.0f,
        std::nextafter(3.0f, 0.0f),
        std::nextafter(3.0f, 4.0f),
        3.0f,
    };
    for (float coordinate : edgeCoordinates) {
        const Ray edgeRay{glm::vec3(coordinate, 2.0f, coordinate),
                          glm::vec3(0.0f, -2.0f, 0.0f)};
        Hit brute{};
        Hit accelerated{};
        const bool bruteFound = closestHit(view, edgeRay, 0.0f, 10.0f, false, brute);
        const bool bvhFound = closestHit(view, edgeRay, 0.0f, 10.0f, true, accelerated);
        CHECK(bruteFound == bvhFound);
        CHECK(anyHit(view, edgeRay, 0.0f, 10.0f, false) ==
              anyHit(view, edgeRay, 0.0f, 10.0f, true));
        if (bruteFound) {
            checkHitsExactlyEqual(accelerated, brute);
        }
    }

    // Transforming both the ray and an analytic primitive must preserve its
    // original ray parameter, even under rotation, reflection, and nonuniform scale.
    for (std::size_t primitive = 4; primitive < 20; ++primitive) {
        const Geom& geom = geoms[primitive];
        const glm::vec3 localOrigin(0.0f, 0.0f, -2.0f);
        const glm::vec3 localDirection(0.0f, 0.0f, 3.0f);
        const Ray transformedRay{
            glm::vec3(geom.transform * glm::vec4(localOrigin, 1.0f)),
            glm::vec3(geom.transform * glm::vec4(localDirection, 0.0f))};
        const std::vector<Geom> oneGeom{geom};
        std::vector<BvhNode> oneNode;
        std::vector<int> oneIndex;
        buildBvh(oneGeom, oneNode, oneIndex);
        const SceneView oneView = makeView(oneGeom, materials, oneNode, oneIndex);
        Hit hit{};
        CHECK(closestHit(oneView, transformedRay, 0.0f, 2.0f, true, hit));
        CHECK_NEAR(hit.t, 0.5f, 3.0e-5f);
    }

    std::uint32_t randomState = 0x5650b17u;
    const float magnitudes[] = {0.001f, 0.1f, 1.0f, 10.0f, 1000.0f};
    for (int sample = 0; sample < 12000; ++sample) {
        Ray ray{};
        ray.origin = glm::vec3(10.0f * signedRandom(randomState),
                               8.0f * signedRandom(randomState) + 2.0f,
                               10.0f * signedRandom(randomState));
        glm::vec3 direction;
        if (sample % 3 == 0) {
            const Geom& target = geoms[static_cast<std::size_t>(
                advanceRandom(randomState) % static_cast<std::uint32_t>(geoms.size()))];
            const glm::vec3 targetPoint = target.type == TRIANGLE
                ? (target.vertices[0] + target.vertices[1] + target.vertices[2]) / 3.0f
                : target.translation;
            direction = targetPoint - ray.origin;
        } else {
            direction = glm::vec3(signedRandom(randomState), signedRandom(randomState),
                                  signedRandom(randomState));
        }
        if (glm::dot(direction, direction) == 0.0f) {
            direction.z = 1.0f;
        }
        ray.direction = direction * magnitudes[sample % 5];
        const float tMin = (sample % 7 == 0) ? 0.0f : 1.0e-4f;
        const float tMax = (sample % 4 == 0)
            ? 0.05f + 30.0f * (0.5f * signedRandom(randomState) + 0.5f)
            : 10000.0f;

        Hit brute{};
        Hit accelerated{};
        const bool bruteFound = closestHit(view, ray, tMin, tMax, false, brute);
        const bool bvhFound = closestHit(view, ray, tMin, tMax, true, accelerated);
        CHECK(bruteFound == bvhFound);
        CHECK(anyHit(view, ray, tMin, tMax, false) ==
              anyHit(view, ray, tMin, tMax, true));
        if (bruteFound) {
            checkHitsExactlyEqual(accelerated, brute);
        }
    }
}

void compareSceneRayExactly(const SceneView& view, const Ray& ray, float tMin, float tMax,
                            Hit* returnedHit = nullptr) {
    Hit brute{};
    Hit accelerated{};
    const bool bruteFound = closestHit(view, ray, tMin, tMax, false, brute);
    const bool bvhFound = closestHit(view, ray, tMin, tMax, true, accelerated);
    CHECK(bruteFound == bvhFound);
    CHECK(anyHit(view, ray, tMin, tMax, false) ==
          anyHit(view, ray, tMin, tMax, true));
    if (bruteFound) {
        checkHitsExactlyEqual(accelerated, brute);
        if (returnedHit != nullptr) {
            *returnedHit = brute;
        }
    } else if (returnedHit != nullptr) {
        returnedHit->primitiveId = -1;
    }
}

void validateRepositoryScene(const fs::path& path) {
    Scene scene(path.string());
    const SceneView view = scene.view();
    CHECK(view.geomCount == static_cast<int>(scene.geoms.size()));
    CHECK(view.nodeCount == static_cast<int>(scene.bvhNodes.size()));
    CHECK(view.textureCount == static_cast<int>(scene.textures.size()));
    CHECK(view.texelCount == static_cast<int>(scene.texels.size()));

    const Camera& camera = scene.state.camera;
    CHECK_NEAR(glm::length(camera.view), 1.0f, 2.0e-5f);
    CHECK_NEAR(glm::length(camera.right), 1.0f, 2.0e-5f);
    CHECK_NEAR(glm::length(camera.up), 1.0f, 2.0e-5f);
    CHECK_NEAR(glm::dot(camera.view, camera.right), 0.0f, 2.0e-5f);
    CHECK_NEAR(glm::dot(camera.view, camera.up), 0.0f, 2.0e-5f);
    CHECK_NEAR(glm::dot(camera.right, camera.up), 0.0f, 2.0e-5f);

    for (const Geom& geom : scene.geoms) {
        if (geom.type != TRIANGLE) {
            continue;
        }
        const glm::vec3 geometric = glm::normalize(
            glm::cross(geom.vertices[1] - geom.vertices[0],
                       geom.vertices[2] - geom.vertices[0]));
        CHECK(finite(geometric));
        if (geom.hasNormals) {
            for (const glm::vec3& normal : geom.normals) {
                CHECK(glm::dot(normal, geometric) >= -1.0e-6f);
            }
        }
    }

    for (const Material& material : scene.materials) {
        CHECK(material.procedural == 0 || material.procedural == 1);
        if (material.bumpTextureId >= 0) {
            CHECK_NEAR(material.bumpStrength, 0.025f, 1.0e-7f);
        }
    }
    for (const TextureDescriptor& descriptor : scene.textures) {
        CHECK(descriptor.width > 0);
        CHECK(descriptor.height > 0);
        CHECK(descriptor.offset >= 0);
        const std::uint64_t end = static_cast<std::uint64_t>(descriptor.offset) +
                                  static_cast<std::uint64_t>(descriptor.width) *
                                      static_cast<std::uint64_t>(descriptor.height);
        CHECK(end <= scene.texels.size());
    }
    for (const glm::vec3& texel : scene.texels) {
        CHECK(texel.r >= 0.0f && texel.r <= 1.0f);
        CHECK(texel.g >= 0.0f && texel.g <= 1.0f);
        CHECK(texel.b >= 0.0f && texel.b <= 1.0f);
    }
    for (const std::string& asset : scene.assets) {
        CHECK(fs::path(asset).is_absolute());
        CHECK(fs::path(asset) == fs::canonical(asset));
    }

    int primarySamples = 64;
    const std::string filename = path.filename().string();
    if (filename == "optical-workbench.json") {
        primarySamples = 2048;
    } else if (filename == "light-test-sphere.json" ||
               filename == "light-test-cube.json" ||
               filename == "light-test-mesh.json") {
        primarySamples = 768;
    }

    std::uint32_t randomState = 0x9e3779b9u;
    for (unsigned char character : filename) {
        randomState = randomState * 33u + character;
    }
    for (int sample = 0; sample < primarySamples; ++sample) {
        const float pixelX = (0.5f * signedRandom(randomState) + 0.5f) *
                             static_cast<float>(camera.resolution.x);
        const float pixelY = (0.5f * signedRandom(randomState) + 0.5f) *
                             static_cast<float>(camera.resolution.y);
        const glm::vec2 offset(pixelX - 0.5f * static_cast<float>(camera.resolution.x),
                               pixelY - 0.5f * static_cast<float>(camera.resolution.y));
        Ray primary{};
        primary.origin = camera.position;
        primary.direction = camera.view -
                            camera.right * (camera.pixelLength.x * offset.x) -
                            camera.up * (camera.pixelLength.y * offset.y);
        primary.direction *= (sample % 3 == 0) ? 0.125f :
                             ((sample % 3 == 1) ? 1.0f : 8.0f);
        Hit primaryHit{};
        compareSceneRayExactly(view, primary, 0.0f, 100000.0f, &primaryHit);

        if (primaryHit.primitiveId >= 0) {
            Ray secondary{};
            secondary.origin = primaryHit.position + primaryHit.geometricNormal * 1.0e-4f;
            secondary.direction = glm::vec3(signedRandom(randomState),
                                            signedRandom(randomState),
                                            signedRandom(randomState));
            if (glm::dot(secondary.direction, secondary.direction) == 0.0f) {
                secondary.direction = primaryHit.geometricNormal;
            }
            const float shadowMaximum = 0.05f +
                30.0f * (0.5f * signedRandom(randomState) + 0.5f);
            compareSceneRayExactly(view, secondary, 1.0e-4f, shadowMaximum);
        }
    }
}

using TestFunction = void (*)();

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::pair<const char*, TestFunction>> tests = {
        {"POD defaults", testPodDefaults},
        {"OBJ index forms and independent negative indices",
         testObjAllIndexFormsAndIndependentNegativeIndices},
        {"OBJ convex quad fan, comments, and organizational records",
         testObjConvexQuadFanCommentsAndOrganizationalRecords},
        {"OBJ invalid geometry and syntax", testObjInvalidGeometryAndSyntax},
        {"OBJ singular transform and missing file", testObjSingularTransformAndMissingFile},
        {"OBJ mirrored transform", testObjMirroredTransformPreservesWindingNormalsAndAttributes},
        {"scene materials, camera, and view", testSceneMaterialsCameraAndView},
        {"camera lens and validation", testCameraExplicitLensAndValidationErrors},
        {"scene unknown and malformed inputs", testSceneUnknownAndMalformedInputs},
        {"texture color/bump/orientation/assets", testTextureLoadingColorBumpOrientationAndAssets},
        {"texture failures", testTextureFailures},
        {"primitive intersections and BVH agreement", testPrimitiveIntersectionsAndBvhAgreement},
        {"triangle barycentrics/normals/derivatives", testTriangleBarycentricsNormalsAndDerivatives},
        {"stable closest tie break", testStableClosestTieBreak},
        {"BVH construction/determinism/depth cap", testBvhConstructionDeterminismMedianAndDepthCap},
        {"empty geometry and invalid BVH fallback", testEmptyGeometryAndInvalidBvhFallback},
        {"BVH depth/stack overflow fallback", testBvhDepthOrStackOverflowFallsBack},
        {"randomized transformed BVH equivalence", testRandomizedTransformedBvhEquivalence},
    };

    int failed = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test.first << ": unknown exception\n";
        }
    }

    for (int argument = 1; argument < argc; ++argument) {
        try {
            validateRepositoryScene(fs::path(argv[argument]));
            std::cout << "[SCENE PASS] " << argv[argument] << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[SCENE FAIL] " << argv[argument] << ": " << error.what() << '\n';
        }
    }

    std::cout << "Executed " << tests.size() << " test groups, " << (argc - 1)
              << " repository scene(s), and " << gAssertions
              << " assertions; " << failed << " group(s) failed.\n";
    return failed == 0 ? 0 : 1;
}
