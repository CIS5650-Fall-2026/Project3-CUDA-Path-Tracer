#include "checkpoint.h"
#include "geometry.cuh"
#include "scene.h"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

class AssertionFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

int gAssertions = 0;

[[noreturn]] void fail(const char* file, int line, const std::string& message) {
    std::ostringstream output;
    output << file << ':' << line << ": " << message;
    throw AssertionFailure(output.str());
}

void check(bool condition, const char* expression, const char* file, int line) {
    ++gAssertions;
    if (!condition) {
        fail(file, line, std::string("check failed: ") + expression);
    }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

void checkNear(float actual, float expected, float tolerance, const char* actualExpression,
               const char* expectedExpression, const char* file, int line) {
    ++gAssertions;
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        std::ostringstream output;
        output << std::setprecision(9) << actualExpression << " (" << actual
               << ") differs from " << expectedExpression << " (" << expected
               << ") by more than " << tolerance;
        fail(file, line, output.str());
    }
}

#define CHECK_NEAR(actual, expected, tolerance) \
    checkNear((actual), (expected), (tolerance), #actual, #expected, __FILE__, __LINE__)

void checkVec2(const glm::vec2& actual, const glm::vec2& expected, float tolerance,
               const char* file, int line) {
    checkNear(actual.x, expected.x, tolerance, "actual.x", "expected.x", file, line);
    checkNear(actual.y, expected.y, tolerance, "actual.y", "expected.y", file, line);
}

void checkVec3(const glm::vec3& actual, const glm::vec3& expected, float tolerance,
               const char* file, int line) {
    checkNear(actual.x, expected.x, tolerance, "actual.x", "expected.x", file, line);
    checkNear(actual.y, expected.y, tolerance, "actual.y", "expected.y", file, line);
    checkNear(actual.z, expected.z, tolerance, "actual.z", "expected.z", file, line);
}

#define CHECK_VEC2(actual, expected, tolerance) \
    checkVec2((actual), (expected), (tolerance), __FILE__, __LINE__)
#define CHECK_VEC3(actual, expected, tolerance) \
    checkVec3((actual), (expected), (tolerance), __FILE__, __LINE__)

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = fs::temp_directory_path() /
                    ("cis5650_gltf_tests_" + std::to_string(ticks) + "_" +
                     std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path_, error)) {
                return;
            }
        }
        throw std::runtime_error("could not create a unique glTF test directory");
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

using Bytes = std::vector<std::uint8_t>;

void writeBytes(const fs::path& path, const Bytes& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not create fixture '" + path.string() + "'");
    }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("could not write fixture '" + path.string() + "'");
    }
}

void writeText(const fs::path& path, const std::string& text) {
    writeBytes(path, Bytes(text.begin(), text.end()));
}

void appendU16(Bytes& output, std::uint16_t value) {
    output.push_back(static_cast<std::uint8_t>(value));
    output.push_back(static_cast<std::uint8_t>(value >> 8u));
}

void appendU32(Bytes& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value));
    output.push_back(static_cast<std::uint8_t>(value >> 8u));
    output.push_back(static_cast<std::uint8_t>(value >> 16u));
    output.push_back(static_cast<std::uint8_t>(value >> 24u));
}

void appendF32(Bytes& output, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "tests require binary32 floats");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(output, bits);
}

void overwriteU32(Bytes& output, std::size_t offset, std::uint32_t value) {
    CHECK(offset <= output.size());
    CHECK(output.size() - offset >= 4u);
    output[offset + 0] = static_cast<std::uint8_t>(value);
    output[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
    output[offset + 2] = static_cast<std::uint8_t>(value >> 16u);
    output[offset + 3] = static_cast<std::uint8_t>(value >> 24u);
}

void overwriteF32(Bytes& output, std::size_t offset, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    overwriteU32(output, offset, bits);
}

void align(Bytes& output, std::size_t alignment) {
    while (output.size() % alignment != 0) {
        output.push_back(0);
    }
}

std::string base64(const Bytes& input) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((input.size() + 2u) / 3u) * 4u);
    for (std::size_t offset = 0; offset < input.size(); offset += 3) {
        const std::uint32_t first = input[offset];
        const std::uint32_t second = offset + 1 < input.size() ? input[offset + 1] : 0;
        const std::uint32_t third = offset + 2 < input.size() ? input[offset + 2] : 0;
        const std::uint32_t packed = (first << 16u) | (second << 8u) | third;
        result.push_back(alphabet[(packed >> 18u) & 63u]);
        result.push_back(alphabet[(packed >> 12u) & 63u]);
        result.push_back(offset + 1 < input.size() ? alphabet[(packed >> 6u) & 63u] : '=');
        result.push_back(offset + 2 < input.size() ? alphabet[packed & 63u] : '=');
    }
    return result;
}

Bytes makeBmp1x1(std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    Bytes bytes;
    bytes.reserve(58);
    bytes.push_back('B');
    bytes.push_back('M');
    appendU32(bytes, 58);
    appendU16(bytes, 0);
    appendU16(bytes, 0);
    appendU32(bytes, 54);
    appendU32(bytes, 40);
    appendU32(bytes, 1);
    appendU32(bytes, 1);
    appendU16(bytes, 1);
    appendU16(bytes, 24);
    appendU32(bytes, 0);
    appendU32(bytes, 4);
    appendU32(bytes, 2835);
    appendU32(bytes, 2835);
    appendU32(bytes, 0);
    appendU32(bytes, 0);
    bytes.push_back(blue);
    bytes.push_back(green);
    bytes.push_back(red);
    bytes.push_back(0);
    CHECK(bytes.size() == 58u);
    return bytes;
}

enum class IndexKind { None, U8, U16, U32 };
enum class UvKind { None, Float, NormalizedU8, NormalizedU16 };

struct TriangleFixture {
    json model;
    Bytes bytes;
    int positionAccessor = -1;
    int normalAccessor = -1;
    int uvAccessor = -1;
    int indexAccessor = -1;
    int positionView = -1;
    int normalView = -1;
    int uvView = -1;
    int indexView = -1;
    std::size_t firstPositionByte = 0;
};

int addView(TriangleFixture& fixture, std::size_t offset, std::size_t length,
            std::size_t stride = 0) {
    json view = {{"buffer", 0}, {"byteOffset", offset}, {"byteLength", length}};
    if (stride != 0) {
        view["byteStride"] = stride;
    }
    fixture.model["bufferViews"].push_back(std::move(view));
    return static_cast<int>(fixture.model["bufferViews"].size() - 1u);
}

int addAccessor(TriangleFixture& fixture, int view, int componentType,
                const char* type, std::size_t count, std::size_t byteOffset = 0,
                bool normalized = false) {
    json accessor = {{"bufferView", view},
                     {"byteOffset", byteOffset},
                     {"componentType", componentType},
                     {"count", count},
                     {"type", type}};
    if (normalized) {
        accessor["normalized"] = true;
    }
    fixture.model["accessors"].push_back(std::move(accessor));
    return static_cast<int>(fixture.model["accessors"].size() - 1u);
}

TriangleFixture makeTriangle(IndexKind indexKind = IndexKind::U8,
                             UvKind uvKind = UvKind::NormalizedU8,
                             bool includeNormals = true,
                             bool stridedPositions = true) {
    TriangleFixture fixture;
    fixture.model = {
        {"asset", {{"version", "2.0"}, {"generator", "original gltf_tests fixture"}}},
        {"buffers", json::array({{{"byteLength", 0}}})},
        {"bufferViews", json::array()},
        {"accessors", json::array()},
        {"meshes", json::array()},
        {"nodes", json::array({{{"mesh", 0}}})},
        {"scenes", json::array({{{"nodes", json::array({0})}}})},
        {"scene", 0}
    };

    align(fixture.bytes, 4);
    const std::size_t positionViewStart = fixture.bytes.size();
    std::size_t positionAccessorOffset = 0;
    std::size_t positionStride = 0;
    if (stridedPositions) {
        fixture.bytes.insert(fixture.bytes.end(), 4, 0x7bu);
        positionAccessorOffset = 4;
        positionStride = 16;
    }
    fixture.firstPositionByte = fixture.bytes.size();
    const std::array<glm::vec3, 3> positions = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)};
    for (const glm::vec3& position : positions) {
        appendF32(fixture.bytes, position.x);
        appendF32(fixture.bytes, position.y);
        appendF32(fixture.bytes, position.z);
        if (stridedPositions) {
            appendU32(fixture.bytes, 0xdecafbadu);
        }
    }
    fixture.positionView = addView(fixture, positionViewStart,
                                   fixture.bytes.size() - positionViewStart, positionStride);
    fixture.positionAccessor = addAccessor(fixture, fixture.positionView, 5126, "VEC3", 3,
                                           positionAccessorOffset);

    if (includeNormals) {
        align(fixture.bytes, 4);
        const std::size_t start = fixture.bytes.size();
        for (int vertex = 0; vertex < 3; ++vertex) {
            appendF32(fixture.bytes, 0.0f);
            appendF32(fixture.bytes, 0.0f);
            appendF32(fixture.bytes, 1.0f);
        }
        fixture.normalView = addView(fixture, start, fixture.bytes.size() - start);
        fixture.normalAccessor = addAccessor(fixture, fixture.normalView, 5126, "VEC3", 3);
    }

    if (uvKind != UvKind::None) {
        align(fixture.bytes, uvKind == UvKind::NormalizedU8 ? 1 :
                             (uvKind == UvKind::NormalizedU16 ? 2 : 4));
        const std::size_t start = fixture.bytes.size();
        int componentType = 5126;
        bool normalized = false;
        std::size_t stride = 0;
        if (uvKind == UvKind::Float) {
            appendF32(fixture.bytes, 0.0f); appendF32(fixture.bytes, 0.0f);
            appendF32(fixture.bytes, 1.0f); appendF32(fixture.bytes, 0.25f);
            appendF32(fixture.bytes, 0.25f); appendF32(fixture.bytes, 1.0f);
        } else if (uvKind == UvKind::NormalizedU8) {
            componentType = 5121;
            normalized = true;
            stride = 4;
            fixture.bytes.insert(fixture.bytes.end(), {0, 0, 91, 92,
                                                       255, 64, 93, 94,
                                                       32, 255, 95, 96});
        } else {
            componentType = 5123;
            normalized = true;
            stride = 8;
            appendU16(fixture.bytes, 0); appendU16(fixture.bytes, 0);
            appendU16(fixture.bytes, 77); appendU16(fixture.bytes, 177);
            appendU16(fixture.bytes, 65535); appendU16(fixture.bytes, 32768);
            appendU16(fixture.bytes, 78); appendU16(fixture.bytes, 178);
            appendU16(fixture.bytes, 16384); appendU16(fixture.bytes, 65535);
            appendU16(fixture.bytes, 79); appendU16(fixture.bytes, 179);
        }
        fixture.uvView = addView(fixture, start, fixture.bytes.size() - start, stride);
        fixture.uvAccessor = addAccessor(fixture, fixture.uvView, componentType, "VEC2", 3,
                                         0, normalized);
    }

    if (indexKind != IndexKind::None) {
        const std::size_t alignment = indexKind == IndexKind::U32 ? 4 :
                                      (indexKind == IndexKind::U16 ? 2 : 1);
        align(fixture.bytes, alignment);
        const std::size_t start = fixture.bytes.size();
        int componentType = 5121;
        if (indexKind == IndexKind::U8) {
            fixture.bytes.insert(fixture.bytes.end(), {0, 1, 2});
        } else if (indexKind == IndexKind::U16) {
            componentType = 5123;
            appendU16(fixture.bytes, 0); appendU16(fixture.bytes, 1); appendU16(fixture.bytes, 2);
        } else {
            componentType = 5125;
            appendU32(fixture.bytes, 0); appendU32(fixture.bytes, 1); appendU32(fixture.bytes, 2);
        }
        fixture.indexView = addView(fixture, start, fixture.bytes.size() - start);
        fixture.indexAccessor = addAccessor(fixture, fixture.indexView, componentType,
                                            "SCALAR", 3);
    }

    json attributes = {{"POSITION", fixture.positionAccessor}};
    if (fixture.normalAccessor >= 0) {
        attributes["NORMAL"] = fixture.normalAccessor;
    }
    if (fixture.uvAccessor >= 0) {
        attributes["TEXCOORD_0"] = fixture.uvAccessor;
    }
    json primitive = {{"attributes", std::move(attributes)}};
    if (fixture.indexAccessor >= 0) {
        primitive["indices"] = fixture.indexAccessor;
    }
    fixture.model["meshes"].push_back({{"primitives", json::array({primitive})}});
    fixture.model["buffers"][0]["byteLength"] = fixture.bytes.size();
    return fixture;
}

enum class Container { External, DataUri, Glb };

void writeGlb(const fs::path& path, const json& sourceModel, const Bytes& binary,
              std::size_t extraBinaryPadding = 0, bool includeBinaryChunk = true,
              bool bindFirstBufferToBinaryChunk = true) {
    CHECK(extraBinaryPadding % 4u == 0u);
    json model = sourceModel;
    if (bindFirstBufferToBinaryChunk) {
        model["buffers"][0].erase("uri");
    }
    std::string jsonText = model.dump();
    while (jsonText.size() % 4u != 0) {
        jsonText.push_back(' ');
    }
    Bytes paddedBinary = binary;
    while (paddedBinary.size() % 4u != 0) {
        paddedBinary.push_back(0);
    }
    paddedBinary.insert(paddedBinary.end(), extraBinaryPadding, 0);
    const std::uint64_t total64 = 12u + 8u + jsonText.size() +
                                  (includeBinaryChunk ? 8u + paddedBinary.size() : 0u);
    CHECK(total64 <= std::numeric_limits<std::uint32_t>::max());
    Bytes output;
    appendU32(output, 0x46546c67u);
    appendU32(output, 2u);
    appendU32(output, static_cast<std::uint32_t>(total64));
    appendU32(output, static_cast<std::uint32_t>(jsonText.size()));
    appendU32(output, 0x4e4f534au);
    output.insert(output.end(), jsonText.begin(), jsonText.end());
    if (includeBinaryChunk) {
        appendU32(output, static_cast<std::uint32_t>(paddedBinary.size()));
        appendU32(output, 0x004e4942u);
        output.insert(output.end(), paddedBinary.begin(), paddedBinary.end());
    }
    writeBytes(path, output);
}

json cameraJson() {
    return {{"RES", json::array({32, 24})},
            {"FOVY", 45.0},
            {"ITERATIONS", 1},
            {"DEPTH", 2},
            {"FILE", "gltf-test-output"},
            {"EYE", json::array({0, 0, 8})},
            {"LOOKAT", json::array({0, 0, 0})},
            {"UP", json::array({0, 1, 0})}};
}

struct WrittenFixture {
    fs::path scene;
    fs::path model;
    fs::path binary;
};

WrittenFixture writeFixture(const TemporaryDirectory& temporary, const std::string& stem,
                            TriangleFixture fixture, Container container,
                            const json& objectOverrides = json::object(),
                            bool writeExternalBuffer = true,
                            const std::string& externalUri = std::string()) {
    WrittenFixture result;
    result.model = temporary.path() / (stem + (container == Container::Glb ? ".glb" : ".gltf"));
    result.scene = temporary.path() / (stem + ".scene.json");

    if (container == Container::External) {
        const std::string uri = externalUri.empty() ? stem + ".bin" : externalUri;
        fixture.model["buffers"][0]["uri"] = uri;
        std::string decodedName = uri;
        for (std::size_t position = 0; (position = decodedName.find("%20", position)) !=
                                       std::string::npos;) {
            decodedName.replace(position, 3, " ");
            ++position;
        }
        result.binary = temporary.path() / decodedName;
        if (writeExternalBuffer) {
            writeBytes(result.binary, fixture.bytes);
        }
        writeText(result.model, fixture.model.dump(2));
    } else if (container == Container::DataUri) {
        fixture.model["buffers"][0]["uri"] =
            "data:application/octet-stream;base64," + base64(fixture.bytes);
        writeText(result.model, fixture.model.dump(2));
    } else {
        writeGlb(result.model, fixture.model, fixture.bytes);
    }

    json object = {{"TYPE", "gltf"},
                   {"FILE", result.model.filename().generic_string()},
                   {"MATERIAL", "override"},
                   {"TRANS", json::array({0, 0, 0})},
                   {"ROTAT", json::array({0, 0, 0})},
                   {"SCALE", json::array({1, 1, 1})}};
    for (const auto& entry : objectOverrides.items()) {
        object[entry.key()] = entry.value();
    }
    json scene = {
        {"Materials", {{"override", {{"TYPE", "Diffuse"},
                                      {"RGB", json::array({0.125, 0.25, 0.5})}}}}},
        {"Objects", json::array({object})},
        {"Camera", cameraJson()}
    };
    writeText(result.scene, scene.dump(2));
    return result;
}

Scene loadFixture(const TemporaryDirectory& temporary, const std::string& stem,
                  TriangleFixture fixture, Container container,
                  const json& objectOverrides = json::object(),
                  WrittenFixture* written = nullptr,
                  const std::string& externalUri = std::string()) {
    const WrittenFixture files = writeFixture(temporary, stem, std::move(fixture), container,
                                               objectOverrides, true, externalUri);
    if (written != nullptr) {
        *written = files;
    }
    return Scene(files.scene.string());
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return text;
}

template <typename Function>
void expectRuntimeError(Function&& function, std::initializer_list<const char*> needles) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        ++gAssertions;
        const std::string message = lower(error.what());
        bool semantic = needles.size() == 0;
        for (const char* needle : needles) {
            semantic = semantic || message.find(lower(needle)) != std::string::npos;
        }
        if (!semantic) {
            std::ostringstream output;
            output << "diagnostic '" << error.what() << "' did not contain any expected term";
            fail(__FILE__, __LINE__, output.str());
        }
        return;
    } catch (const std::exception& error) {
        fail(__FILE__, __LINE__, std::string("expected runtime_error, got: ") + error.what());
    }
    fail(__FILE__, __LINE__, "expected a runtime_error, but loading succeeded");
}

void expectRejected(const TemporaryDirectory& temporary, const std::string& stem,
                    TriangleFixture fixture, std::initializer_list<const char*> needles,
                    Container container = Container::External) {
    const WrittenFixture files = writeFixture(temporary, stem, std::move(fixture), container);
    expectRuntimeError([&] { Scene scene(files.scene.string()); }, needles);
}

bool containsAsset(const Scene& scene, const fs::path& path) {
    const std::string canonical = fs::canonical(path).string();
    return std::find(scene.assets.begin(), scene.assets.end(), canonical) != scene.assets.end();
}

void checkTriangle(const Geom& triangle) {
    CHECK(triangle.type == TRIANGLE);
    CHECK(triangle.materialid == 0);
    CHECK_VEC3(triangle.vertices[0], glm::vec3(0.0f, 0.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3(triangle.vertices[1], glm::vec3(1.0f, 0.0f, 0.0f), 1.0e-6f);
    CHECK_VEC3(triangle.vertices[2], glm::vec3(0.0f, 1.0f, 0.0f), 1.0e-6f);
}

void checkDeterministicGeom(const Geom& first, const Geom& second) {
    CHECK(first.type == second.type);
    CHECK(first.materialid == second.materialid);
    CHECK(first.hasNormals == second.hasNormals);
    CHECK(first.hasUVs == second.hasUVs);
    CHECK_VEC3(first.motionTranslation, second.motionTranslation, 0.0f);
    for (int corner = 0; corner < 3; ++corner) {
        CHECK_VEC3(first.vertices[corner], second.vertices[corner], 0.0f);
        CHECK_VEC3(first.normals[corner], second.normals[corner], 0.0f);
        CHECK_VEC2(first.uvs[corner], second.uvs[corner], 0.0f);
    }
}

void testExternalPercentUriAttributesMotionAssetsAndDeterminism() {
    TemporaryDirectory temporary;
    TriangleFixture fixture = makeTriangle(IndexKind::U8, UvKind::NormalizedU8, true, true);
    fixture.model["materials"] = json::array({{{"name", "ignored native material"}}});
    fixture.model["meshes"][0]["primitives"][0]["material"] = 0;
    fixture.model["images"] = json::array({{{"uri", "image%20asset.bmp"}},
                                            {{"uri", "image%20asset.bmp"}}});
    writeBytes(temporary.path() / "image asset.bmp", makeBmp1x1(21, 43, 87));

    const json object = {{"MOTION_TRANSLATION", json::array({3, -2, 1})}};
    WrittenFixture files;
    Scene scene = loadFixture(temporary, "percent", fixture, Container::External,
                              object, &files, "mesh%20data.bin");
    CHECK(scene.geoms.size() == 1u);
    CHECK(scene.materials.size() == 1u);
    const Geom& triangle = scene.geoms[0];
    checkTriangle(triangle);
    CHECK(triangle.hasNormals);
    CHECK(triangle.hasUVs);
    CHECK_VEC3(triangle.motionTranslation, glm::vec3(3.0f, -2.0f, 1.0f), 0.0f);
    for (int corner = 0; corner < 3; ++corner) {
        CHECK_VEC3(triangle.normals[corner], glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);
    }
    CHECK_VEC2(triangle.uvs[0], glm::vec2(0.0f, 1.0f), 1.0e-6f);
    CHECK_VEC2(triangle.uvs[1], glm::vec2(1.0f, 1.0f - 64.0f / 255.0f), 1.0e-6f);
    CHECK_VEC2(triangle.uvs[2], glm::vec2(32.0f / 255.0f, 0.0f), 1.0e-6f);
    CHECK_NEAR(scene.materials[0].color.r, 0.125f, 0.0f);
    CHECK_NEAR(scene.materials[0].color.g, 0.25f, 0.0f);
    CHECK_NEAR(scene.materials[0].color.b, 0.5f, 0.0f);

    const fs::path image = temporary.path() / "image asset.bmp";
    CHECK(containsAsset(scene, files.scene));
    CHECK(containsAsset(scene, files.model));
    CHECK(containsAsset(scene, files.binary));
    CHECK(containsAsset(scene, image));
    CHECK(scene.assets.size() == 4u);

    std::uint64_t before = 0;
    std::uint64_t after = 0;
    CHECK(checkpoint::fingerprintFile(image, before).ok());
    writeBytes(image, makeBmp1x1(22, 43, 87));
    CHECK(checkpoint::fingerprintFile(image, after).ok());
    CHECK(before != after);

    std::uint64_t binaryBefore = 0;
    std::uint64_t binaryAfter = 0;
    CHECK(fixture.firstPositionByte > 0u);
    CHECK(checkpoint::fingerprintFile(files.binary, binaryBefore).ok());
    Bytes binaryWithChangedPadding = fixture.bytes;
    binaryWithChangedPadding[0] ^= 0x5au;
    writeBytes(files.binary, binaryWithChangedPadding);
    CHECK(checkpoint::fingerprintFile(files.binary, binaryAfter).ok());
    CHECK(binaryBefore != binaryAfter);

    Scene reloaded(files.scene.string());
    CHECK(reloaded.geoms.size() == scene.geoms.size());
    checkDeterministicGeom(scene.geoms[0], reloaded.geoms[0]);

    Ray movedRay{};
    movedRay.origin = glm::vec3(3.2f, -1.8f, 3.0f);
    movedRay.direction = glm::vec3(0.0f, 0.0f, -1.0f);
    movedRay.time = 1.0f;
    Hit movedHit{};
    CHECK(closestHit(scene.view(), movedRay, 0.0f, 10.0f, true, movedHit));
    CHECK(movedHit.materialId == 0);
    CHECK_NEAR(movedHit.position.z, 1.0f, 1.0e-5f);
}

void testDataUriAndGlbIndexWidthsAndLockedUvFlip() {
    TemporaryDirectory temporary;
    Scene embedded = loadFixture(temporary, "embedded-u16",
                                 makeTriangle(IndexKind::U16, UvKind::NormalizedU16, true, false),
                                 Container::DataUri);
    CHECK(embedded.geoms.size() == 1u);
    checkTriangle(embedded.geoms[0]);
    CHECK_VEC2(embedded.geoms[0].uvs[0], glm::vec2(0.0f, 1.0f), 1.0e-6f);
    CHECK_VEC2(embedded.geoms[0].uvs[1],
               glm::vec2(1.0f, 1.0f - 32768.0f / 65535.0f), 1.0e-6f);
    CHECK_VEC2(embedded.geoms[0].uvs[2],
               glm::vec2(16384.0f / 65535.0f, 0.0f), 1.0e-6f);
    CHECK(embedded.assets.size() == 2u);

    Scene binary = loadFixture(temporary, "binary-u32",
                               makeTriangle(IndexKind::U32, UvKind::Float, true, false),
                               Container::Glb);
    CHECK(binary.geoms.size() == 1u);
    checkTriangle(binary.geoms[0]);
    CHECK_VEC2(binary.geoms[0].uvs[0], glm::vec2(0.0f, 1.0f), 1.0e-6f);
    CHECK_VEC2(binary.geoms[0].uvs[1], glm::vec2(1.0f, 0.75f), 1.0e-6f);
    CHECK_VEC2(binary.geoms[0].uvs[2], glm::vec2(0.25f, 0.0f), 1.0e-6f);
    CHECK(binary.assets.size() == 2u);
}

void testUnindexedAndMissingAttributeFallbacks() {
    TemporaryDirectory temporary;
    Scene floatUv = loadFixture(temporary, "unindexed-float-uv",
                                makeTriangle(IndexKind::None, UvKind::Float, false, false),
                                Container::External);
    CHECK(floatUv.geoms.size() == 1u);
    checkTriangle(floatUv.geoms[0]);
    CHECK(!floatUv.geoms[0].hasNormals);
    CHECK(floatUv.geoms[0].hasUVs);
    for (const glm::vec3& normal : floatUv.geoms[0].normals) {
        CHECK_VEC3(normal, glm::vec3(0.0f, 0.0f, 1.0f), 1.0e-6f);
    }

    Scene absent = loadFixture(temporary, "unindexed-no-attrs",
                               makeTriangle(IndexKind::None, UvKind::None, false, false),
                               Container::External);
    CHECK(absent.geoms.size() == 1u);
    checkTriangle(absent.geoms[0]);
    CHECK(!absent.geoms[0].hasNormals);
    CHECK(!absent.geoms[0].hasUVs);
    CHECK_VEC2(absent.geoms[0].uvs[0], glm::vec2(0.0f, 0.0f), 0.0f);
    CHECK_VEC2(absent.geoms[0].uvs[1], glm::vec2(1.0f, 0.0f), 0.0f);
    CHECK_VEC2(absent.geoms[0].uvs[2], glm::vec2(0.0f, 1.0f), 0.0f);
}

void testSceneSelectionAndSceneLessRootFallback() {
    TemporaryDirectory temporary;
    TriangleFixture fixture = makeTriangle(IndexKind::U8, UvKind::None, false, false);
    fixture.model["nodes"] = json::array({{{"mesh", 0}},
                                           {{"mesh", 0}, {"translation", json::array({5, 0, 0})}}});
    fixture.model["scenes"] = json::array({{{"nodes", json::array({0})}},
                                            {{"nodes", json::array({1})}}});
    fixture.model["scene"] = 1;
    Scene selected = loadFixture(temporary, "selected-scene", fixture, Container::External);
    CHECK(selected.geoms.size() == 1u);
    CHECK_VEC3(selected.geoms[0].vertices[0], glm::vec3(5.0f, 0.0f, 0.0f), 1.0e-6f);

    fixture.model.erase("scene");
    Scene first = loadFixture(temporary, "first-scene", fixture, Container::External);
    CHECK(first.geoms.size() == 1u);
    CHECK_VEC3(first.geoms[0].vertices[0], glm::vec3(0.0f), 1.0e-6f);

    fixture.model.erase("scenes");
    fixture.model["nodes"] = json::array({{{"translation", json::array({1, 0, 0})},
                                            {"children", json::array({1})}},
                                           {{"mesh", 0}, {"translation", json::array({2, 0, 0})}},
                                           {{"mesh", 0}, {"translation", json::array({6, 0, 0})}}});
    Scene roots = loadFixture(temporary, "root-fallback", fixture, Container::External);
    CHECK(roots.geoms.size() == 2u);
    std::array<float, 2> firstXs = {roots.geoms[0].vertices[0].x,
                                    roots.geoms[1].vertices[0].x};
    std::sort(firstXs.begin(), firstXs.end());
    CHECK_NEAR(firstXs[0], 3.0f, 1.0e-6f);
    CHECK_NEAR(firstXs[1], 6.0f, 1.0e-6f);
}

void testHierarchyTrsMatrixInstancesMirroringAndNormals() {
    TemporaryDirectory temporary;
    TriangleFixture fixture = makeTriangle(IndexKind::U8, UvKind::None, true, false);
    const double halfSqrt = std::sqrt(0.5);
    const std::size_t normalStart =
        fixture.model["bufferViews"][fixture.normalView]["byteOffset"].get<std::size_t>();
    for (int corner = 0; corner < 3; ++corner) {
        overwriteF32(fixture.bytes, normalStart + static_cast<std::size_t>(corner) * 12u,
                     0.0f);
        overwriteF32(fixture.bytes, normalStart + static_cast<std::size_t>(corner) * 12u + 4u,
                     static_cast<float>(halfSqrt));
        overwriteF32(fixture.bytes, normalStart + static_cast<std::size_t>(corner) * 12u + 8u,
                     static_cast<float>(halfSqrt));
    }
    fixture.model["nodes"] = json::array({
        {{"translation", json::array({1, 2, 3})}, {"children", json::array({1, 2})}},
        {{"mesh", 0},
         {"translation", json::array({2, 0, 0})},
         {"rotation", json::array({0, 0, halfSqrt, halfSqrt})},
         {"scale", json::array({2, 3, 4})}},
        {{"mesh", 0},
         {"matrix", json::array({-1, 0, 0, 0,
                                   0, 2, 0, 0,
                                   0, 0, 1, 0,
                                  -2, 0, 1, 1})}}
    });
    fixture.model["scenes"] = json::array({{{"nodes", json::array({0})}}});
    fixture.model["scene"] = 0;
    const json object = {{"TRANS", json::array({10, 0, 0})},
                         {"MOTION_TRANSLATION", json::array({0.5, 1.5, -2})}};
    Scene scene = loadFixture(temporary, "hierarchy", fixture, Container::External, object);
    CHECK(scene.geoms.size() == 2u);

    const Geom* trs = nullptr;
    const Geom* matrix = nullptr;
    for (const Geom& geom : scene.geoms) {
        CHECK_VEC3(geom.motionTranslation, glm::vec3(0.5f, 1.5f, -2.0f), 0.0f);
        const glm::vec3 geometric = glm::normalize(
            glm::cross(geom.vertices[1] - geom.vertices[0],
                       geom.vertices[2] - geom.vertices[0]));
        CHECK_VEC3(geometric, glm::vec3(0.0f, 0.0f, 1.0f), 2.0e-6f);
        if (geom.vertices[0].x > 11.0f) {
            trs = &geom;
        } else {
            matrix = &geom;
        }
    }
    CHECK(trs != nullptr);
    CHECK(matrix != nullptr);
    CHECK_VEC3(trs->vertices[0], glm::vec3(13.0f, 2.0f, 3.0f), 2.0e-5f);
    CHECK_VEC3(trs->vertices[1], glm::vec3(13.0f, 4.0f, 3.0f), 2.0e-5f);
    CHECK_VEC3(trs->vertices[2], glm::vec3(10.0f, 2.0f, 3.0f), 2.0e-5f);
    CHECK_VEC3(matrix->vertices[0], glm::vec3(9.0f, 2.0f, 4.0f), 1.0e-6f);
    CHECK_VEC3(matrix->vertices[1], glm::vec3(9.0f, 4.0f, 4.0f), 1.0e-6f);
    CHECK_VEC3(matrix->vertices[2], glm::vec3(8.0f, 2.0f, 4.0f), 1.0e-6f);
    for (const glm::vec3& normal : trs->normals) {
        CHECK_VEC3(normal, glm::vec3(-0.8f, 0.0f, 0.6f), 2.0e-6f);
    }
    const float inverseSqrtFive = 1.0f / std::sqrt(5.0f);
    for (const glm::vec3& normal : matrix->normals) {
        CHECK_VEC3(normal, glm::vec3(0.0f, inverseSqrtFive,
                                    2.0f * inverseSqrtFive), 2.0e-6f);
    }

    TriangleFixture objectFixture = makeTriangle(IndexKind::U8, UvKind::None, false, false);
    objectFixture.model["meshes"][0]["primitives"][0]["mode"] = 4;
    const json transformedObject = {{"TRANS", json::array({5, 6, 7})},
                                    {"ROTAT", json::array({0, 0, 90})},
                                    {"SCALE", json::array({2, 3, 4})}};
    Scene transformed = loadFixture(temporary, "object-transform", std::move(objectFixture),
                                    Container::External, transformedObject);
    CHECK(transformed.geoms.size() == 1u);
    CHECK_VEC3(transformed.geoms[0].vertices[0], glm::vec3(5.0f, 6.0f, 7.0f), 2.0e-6f);
    CHECK_VEC3(transformed.geoms[0].vertices[1], glm::vec3(5.0f, 8.0f, 7.0f), 2.0e-6f);
    CHECK_VEC3(transformed.geoms[0].vertices[2], glm::vec3(2.0f, 6.0f, 7.0f), 2.0e-6f);
}

void testUnsupportedGeometryChangingFeatures() {
    TemporaryDirectory temporary;
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["extensionsRequired"] = json::array({"KHR_draco_mesh_compression"});
        expectRejected(temporary, "required-extension", std::move(fixture),
                       {"required extension", "extensionsrequired", "draco"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"].push_back(json::object());
        fixture.model["skins"] = json::array({{{"joints", json::array({1})}}});
        fixture.model["nodes"][0]["skin"] = 0;
        expectRejected(temporary, "skin", std::move(fixture), {"skin"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["meshes"][0]["primitives"][0]["attributes"]["JOINTS_1"] =
            fixture.positionAccessor;
        expectRejected(temporary, "skinning-attribute", std::move(fixture),
                       {"skin", "joints", "attribute"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["animations"] = json::array({{{"channels", json::array()},
                                                     {"samplers", json::array()}}});
        expectRejected(temporary, "animation", std::move(fixture), {"animation"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["meshes"][0]["primitives"][0]["targets"] =
            json::array({{{"POSITION", fixture.positionAccessor}}});
        expectRejected(temporary, "morph-target", std::move(fixture), {"morph", "target"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"][0]["weights"] = json::array({1.0});
        expectRejected(temporary, "morph-weights", std::move(fixture), {"weight", "morph"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["meshes"][0]["primitives"][0]["mode"] = 1;
        expectRejected(temporary, "line-mode", std::move(fixture),
                       {"triangle", "mode", "primitive"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        align(fixture.bytes, 4);
        const std::size_t indexOffset = fixture.bytes.size();
        fixture.bytes.push_back(0);
        const int sparseIndexView = addView(fixture, indexOffset, 1);
        align(fixture.bytes, 4);
        const std::size_t valueOffset = fixture.bytes.size();
        appendF32(fixture.bytes, 9.0f); appendF32(fixture.bytes, 9.0f);
        appendF32(fixture.bytes, 9.0f);
        const int sparseValueView = addView(fixture, valueOffset, 12);
        fixture.model["buffers"][0]["byteLength"] = fixture.bytes.size();
        fixture.model["accessors"][fixture.positionAccessor]["sparse"] = {
            {"count", 1},
            {"indices", {{"bufferView", sparseIndexView}, {"componentType", 5121}}},
            {"values", {{"bufferView", sparseValueView}}}
        };
        expectRejected(temporary, "sparse", std::move(fixture), {"sparse"});
    }
}

void testHierarchyAndTransformRejections() {
    TemporaryDirectory temporary;
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"] = json::array({{{"mesh", 0}, {"children", json::array({1})}},
                                               {{"children", json::array({0})}}});
        fixture.model["scenes"][0]["nodes"] = json::array({0});
        expectRejected(temporary, "cycle", std::move(fixture), {"cycle", "cyclic"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"][0]["children"] = json::array({99});
        expectRejected(temporary, "bad-child", std::move(fixture),
                       {"node", "child", "index"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["scenes"][0]["nodes"] = json::array({99});
        expectRejected(temporary, "bad-root", std::move(fixture), {"scene", "node", "index"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["scene"] = 7;
        expectRejected(temporary, "bad-default-scene", std::move(fixture),
                       {"default", "scene", "index"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["scene"] = -2;
        expectRejected(temporary, "negative-default-scene", std::move(fixture),
                       {"default", "scene", "negative", "index"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["scene"] = -1;
        expectRejected(temporary, "sentinel-default-scene", std::move(fixture),
                       {"scene", "nonnegative", "index"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"][0]["matrix"] =
            json::array({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        fixture.model["nodes"][0]["translation"] = json::array({1, 0, 0});
        expectRejected(temporary, "matrix-and-trs", std::move(fixture),
                       {"matrix", "trs", "translation"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"][0]["rotation"] = json::array({0, 0, 0, 2});
        expectRejected(temporary, "nonunit-quaternion", std::move(fixture),
                       {"quaternion", "rotation", "unit"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"][0]["scale"] = json::array({1, 0, 1});
        expectRejected(temporary, "singular-scale", std::move(fixture),
                       {"singular", "scale", "transform"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"][0]["matrix"] =
            json::array({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        expectRejected(temporary, "singular-matrix", std::move(fixture),
                       {"singular", "matrix", "transform"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["nodes"] = json::array();
        constexpr int depth = 5000;
        for (int index = 0; index < depth; ++index) {
            json node = json::object();
            if (index + 1 < depth) {
                node["children"] = json::array({index + 1});
            } else {
                node["mesh"] = 0;
            }
            fixture.model["nodes"].push_back(std::move(node));
        }
        fixture.model["scenes"][0]["nodes"] = json::array({0});
        expectRejected(temporary, "deep-hierarchy", std::move(fixture),
                       {"depth", "hierarchy", "4096", "limit"});
    }
}

void testAccessorSchemaAndReferenceRejections() {
    TemporaryDirectory temporary;
    auto rejectMutation = [&](const std::string& stem,
                              const std::function<void(TriangleFixture&)>& mutate,
                              std::initializer_list<const char*> needles) {
        TriangleFixture fixture = makeTriangle();
        mutate(fixture);
        expectRejected(temporary, stem, std::move(fixture), needles);
    };

    rejectMutation("position-type", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.positionAccessor]["type"] = "VEC2";
    }, {"position", "vec3"});
    rejectMutation("position-component", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.positionAccessor]["componentType"] = 5123;
    }, {"position", "float", "component"});
    rejectMutation("normal-type", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.normalAccessor]["type"] = "VEC2";
    }, {"normal", "vec3"});
    rejectMutation("normal-component", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.normalAccessor]["componentType"] = 5123;
    }, {"normal", "float", "component"});
    rejectMutation("uv-type", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.uvAccessor]["type"] = "VEC3";
    }, {"texcoord", "vec2", "uv"});
    rejectMutation("uv-not-normalized", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.uvAccessor]["normalized"] = false;
    }, {"texcoord", "normalized", "uv"});
    rejectMutation("u8-uv-attribute-offset", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.uvAccessor]["byteOffset"] = 1;
    }, {"four-byte", "aligned", "texcoord"});
    {
        TriangleFixture fixture = makeTriangle(IndexKind::U8, UvKind::NormalizedU16);
        fixture.model["accessors"][fixture.uvAccessor]["byteOffset"] = 2;
        expectRejected(temporary, "u16-uv-attribute-offset", std::move(fixture),
                       {"four-byte", "aligned", "texcoord"});
    }
    rejectMutation("normal-count", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.normalAccessor]["count"] = 2;
    }, {"normal", "count", "position"});
    rejectMutation("uv-count", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.uvAccessor]["count"] = 2;
    }, {"texcoord", "count", "position", "uv"});
    rejectMutation("index-type", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.indexAccessor]["componentType"] = 5126;
    }, {"index", "unsigned", "component"});
    rejectMutation("index-not-scalar", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.indexAccessor]["type"] = "VEC2";
    }, {"index", "scalar"});
    rejectMutation("index-count", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.indexAccessor]["count"] = 2;
    }, {"multiple", "three", "index", "triangle"});
    rejectMutation("unindexed-count", [](TriangleFixture& fixture) {
        fixture.model["meshes"][0]["primitives"][0].erase("indices");
        fixture.model["accessors"][fixture.positionAccessor]["count"] = 2;
    }, {"multiple", "three", "position", "triangle"});
    rejectMutation("bad-position-accessor", [](TriangleFixture& fixture) {
        fixture.model["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = 999;
    }, {"position", "accessor", "index"});
    rejectMutation("negative-position-accessor", [](TriangleFixture& fixture) {
        fixture.model["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = -2;
    }, {"position", "accessor", "negative", "index"});
    rejectMutation("bad-index-accessor", [](TriangleFixture& fixture) {
        fixture.model["meshes"][0]["primitives"][0]["indices"] = 999;
    }, {"index", "accessor"});
    rejectMutation("negative-index-accessor", [](TriangleFixture& fixture) {
        fixture.model["meshes"][0]["primitives"][0]["indices"] = -2;
    }, {"index", "nonnegative", "accessor"});
    rejectMutation("bad-mesh-index", [](TriangleFixture& fixture) {
        fixture.model["nodes"][0]["mesh"] = 999;
    }, {"mesh", "index", "range"});
    rejectMutation("negative-mesh-index", [](TriangleFixture& fixture) {
        fixture.model["nodes"][0]["mesh"] = -2;
    }, {"mesh", "nonnegative", "index"});
    rejectMutation("bad-buffer-view", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.positionAccessor]["bufferView"] = 999;
    }, {"buffer", "view", "position", "index"});
    rejectMutation("bad-buffer", [](TriangleFixture& fixture) {
        fixture.model["bufferViews"][fixture.positionView]["buffer"] = 999;
    }, {"buffer", "index", "view"});
    rejectMutation("stride-too-small", [](TriangleFixture& fixture) {
        fixture.model["bufferViews"][fixture.positionView]["byteStride"] = 8;
    }, {"stride", "position", "element"});
    rejectMutation("stride-misaligned", [](TriangleFixture& fixture) {
        fixture.model["bufferViews"][fixture.positionView]["byteStride"] = 14;
    }, {"stride", "align", "multiple"});
    rejectMutation("strided-index", [](TriangleFixture& fixture) {
        fixture.model["bufferViews"][fixture.indexView]["byteStride"] = 4;
        const std::size_t start = fixture.model["bufferViews"][fixture.indexView]
            ["byteOffset"].get<std::size_t>();
        fixture.bytes.resize(start + 9u, 0);
        fixture.model["bufferViews"][fixture.indexView]["byteLength"] = 9;
        fixture.model["buffers"][0]["byteLength"] = fixture.bytes.size();
    }, {"index", "stride"});
    rejectMutation("accessor-out-of-view", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.positionAccessor]["byteOffset"] = 40;
    }, {"range", "bounds", "accessor", "view"});
    rejectMutation("view-out-of-buffer", [](TriangleFixture& fixture) {
        fixture.model["bufferViews"][fixture.positionView]["byteLength"] =
            fixture.bytes.size() + 4096u;
    }, {"range", "bounds", "buffer", "view"});
    rejectMutation("offset-overflow", [](TriangleFixture& fixture) {
        fixture.model["bufferViews"][fixture.positionView]["byteOffset"] =
            std::numeric_limits<std::uint64_t>::max() - 7u;
    }, {"overflow", "range", "bounds", "offset"});
    rejectMutation("accessor-offset-overflow", [](TriangleFixture& fixture) {
        fixture.model["accessors"][fixture.positionAccessor]["byteOffset"] =
            std::numeric_limits<std::uint64_t>::max() - 7u;
    }, {"overflow", "range", "bounds", "offset"});
}

void testRawStorageAndGlbBoundaryRejections() {
    TemporaryDirectory temporary;
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["asset"]["minVersion"] = "2.1";
        expectRejected(temporary, "future-min-version", std::move(fixture),
                       {"minversion", "version", "2.0"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["bufferViews"].push_back({{"buffer", 0},
                                                  {"byteOffset", fixture.bytes.size()},
                                                  {"byteLength", 1}});
        expectRejected(temporary, "unused-view-out-of-buffer", std::move(fixture),
                       {"bufferview", "range", "buffer length"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        const int imageView = static_cast<int>(fixture.model["bufferViews"].size());
        fixture.model["bufferViews"].push_back({{"buffer", 0},
                                                  {"byteOffset", fixture.bytes.size() - 1u},
                                                  {"byteLength", 2}});
        fixture.model["images"] = json::array(
            {{{"bufferView", imageView}, {"mimeType", "image/png"}}});
        expectRejected(temporary, "image-view-out-of-buffer", std::move(fixture),
                       {"image", "bufferview", "range", "buffer length"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["accessors"].push_back(
            {{"bufferView", fixture.positionView},
             {"byteOffset", std::numeric_limits<std::uint64_t>::max() - 7u},
             {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}});
        expectRejected(temporary, "unused-accessor-overflow", std::move(fixture),
                       {"accessor", "overflow", "range", "offset"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["accessors"][fixture.positionAccessor]["count"] =
            std::numeric_limits<std::uint64_t>::max();
        expectRejected(temporary, "accessor-count-multiply-overflow", std::move(fixture),
                       {"accessor", "count", "overflow", "range"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model.erase("scenes");
        fixture.model["scene"] = 0;
        expectRejected(temporary, "scene-index-without-scenes", std::move(fixture),
                       {"scene", "range", "absent"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["buffers"].push_back({{"byteLength", 4}});
        expectRejected(temporary, "second-bin-backed-buffer", std::move(fixture),
                       {"buffers[0]", "bin chunk", "uri"}, Container::Glb);
    }
    {
        TriangleFixture fixture = makeTriangle();
        const WrittenFixture files = writeFixture(
            temporary, "oversized-bin-padding", fixture, Container::Glb);
        writeGlb(files.model, fixture.model, fixture.bytes, 4);
        expectRuntimeError([&] { Scene scene(files.scene.string()); },
                           {"padding", "bin chunk", "bytelength"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        const WrittenFixture files = writeFixture(
            temporary, "missing-bin-chunk", fixture, Container::Glb);
        writeGlb(files.model, fixture.model, fixture.bytes, 0, false);
        expectRuntimeError([&] { Scene scene(files.scene.string()); },
                           {"presence", "bin chunk", "uri-less"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        fixture.model["buffers"][0]["uri"] = "external.bin";
        const WrittenFixture files = writeFixture(
            temporary, "unreferenced-bin-chunk", fixture, Container::Glb);
        writeBytes(temporary.path() / "external.bin", fixture.bytes);
        writeGlb(files.model, fixture.model, fixture.bytes, 0, true, false);
        expectRuntimeError([&] { Scene scene(files.scene.string()); },
                           {"presence", "bin chunk", "uri-less"});
    }
}

void testDecodedDataRejections() {
    TemporaryDirectory temporary;
    {
        TriangleFixture fixture = makeTriangle();
        const std::size_t stride = 16;
        overwriteF32(fixture.bytes, fixture.firstPositionByte + stride,
                     std::numeric_limits<float>::infinity());
        expectRejected(temporary, "infinite-position", std::move(fixture),
                       {"finite", "position", "infinite"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        overwriteF32(fixture.bytes, fixture.firstPositionByte + 2u * 16u + 4u,
                     std::numeric_limits<float>::quiet_NaN());
        expectRejected(temporary, "nan-position", std::move(fixture),
                       {"finite", "position", "nan"});
    }
    {
        TriangleFixture fixture = makeTriangle();
        const std::size_t normalStart =
            fixture.model["bufferViews"][fixture.normalView]["byteOffset"].get<std::size_t>();
        overwriteF32(fixture.bytes, normalStart, std::numeric_limits<float>::infinity());
        expectRejected(temporary, "infinite-normal", std::move(fixture),
                       {"finite", "normal", "infinite"});
    }
    {
        TriangleFixture fixture = makeTriangle(IndexKind::U8);
        const std::size_t indexStart =
            fixture.model["bufferViews"][fixture.indexView]["byteOffset"].get<std::size_t>();
        fixture.bytes[indexStart + 2u] = 7;
        expectRejected(temporary, "index-range", std::move(fixture),
                       {"index", "range", "vertex"});
    }
    {
        TriangleFixture fixture = makeTriangle(IndexKind::None, UvKind::None, false, true);
        const std::size_t stride = 16;
        overwriteF32(fixture.bytes, fixture.firstPositionByte + 2u * stride, 2.0f);
        overwriteF32(fixture.bytes, fixture.firstPositionByte + 2u * stride + 4u, 0.0f);
        expectRejected(temporary, "degenerate", std::move(fixture),
                       {"degenerate", "triangle", "area"});
    }
}

void testUriAndFileRejections() {
    TemporaryDirectory temporary;
    auto uriFailure = [&](const std::string& stem, const std::string& uri,
                          std::initializer_list<const char*> needles) {
        TriangleFixture fixture = makeTriangle();
        fixture.model["buffers"][0]["uri"] = uri;
        const fs::path model = temporary.path() / (stem + ".gltf");
        writeText(model, fixture.model.dump(2));
        const fs::path scene = temporary.path() / (stem + ".scene.json");
        json object = {{"TYPE", "gltf"}, {"FILE", model.filename().generic_string()},
                       {"MATERIAL", "override"}, {"TRANS", json::array({0, 0, 0})},
                       {"ROTAT", json::array({0, 0, 0})}, {"SCALE", json::array({1, 1, 1})}};
        json root = {{"Materials", {{"override", {{"TYPE", "Diffuse"},
                                                    {"RGB", json::array({1, 1, 1})}}}}},
                     {"Objects", json::array({object})}, {"Camera", cameraJson()}};
        writeText(scene, root.dump(2));
        expectRuntimeError([&] { Scene loaded(scene.string()); }, needles);
    };
    uriFailure("remote-http", "https://example.invalid/mesh.bin",
               {"remote", "scheme", "http", "uri"});
    uriFailure("file-uri", "file:///tmp/mesh.bin", {"file uri", "scheme", "uri"});
    uriFailure("bad-percent", "bad%ZZ.bin", {"percent", "uri", "encoding"});
    uriFailure("nul-percent", "bad%00name.bin", {"nul", "null", "uri", "percent"});
    uriFailure("invalid-data", "data:application/octet-stream;base64,!!!!",
               {"data uri", "base64", "decode"});

    TriangleFixture missing = makeTriangle();
    const WrittenFixture files = writeFixture(temporary, "missing-external", std::move(missing),
                                               Container::External, json::object(), false);
    expectRuntimeError([&] { Scene scene(files.scene.string()); },
                       {"buffer", "open", "missing", "resolve"});
}

using TestFunction = void (*)();

} // namespace

int main() {
    const std::vector<std::pair<const char*, TestFunction>> tests = {
        {"external percent URI, attributes, motion, assets, determinism",
         testExternalPercentUriAttributesMotionAssetsAndDeterminism},
        {"data URI and GLB index widths and UV flip",
         testDataUriAndGlbIndexWidthsAndLockedUvFlip},
        {"unindexed and missing attribute fallbacks", testUnindexedAndMissingAttributeFallbacks},
        {"scene selection and scene-less roots", testSceneSelectionAndSceneLessRootFallback},
        {"hierarchy TRS/matrix instances and mirrored normals",
         testHierarchyTrsMatrixInstancesMirroringAndNormals},
        {"unsupported geometry-changing features", testUnsupportedGeometryChangingFeatures},
        {"hierarchy and transform rejections", testHierarchyAndTransformRejections},
        {"accessor schema/reference rejections", testAccessorSchemaAndReferenceRejections},
        {"raw storage and GLB boundary rejections", testRawStorageAndGlbBoundaryRejections},
        {"decoded data rejections", testDecodedDataRejections},
        {"URI and file rejections", testUriAndFileRejections},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.first << ": unknown exception\n";
        }
    }
    std::cout << "Executed " << tests.size() << " glTF test groups and " << gAssertions
              << " assertions; " << failures << " group(s) failed.\n";
    return failures == 0 ? 0 : 1;
}
