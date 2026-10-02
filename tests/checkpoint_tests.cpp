#include "checkpoint.h"
#include "scene.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using checkpoint::CheckpointData;

std::size_t gChecks = 0;

void require(bool condition, const std::string& message) {
    ++gChecks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool sameFloat(float a, float b) {
    std::uint32_t aBits = 0;
    std::uint32_t bBits = 0;
    std::memcpy(&aBits, &a, sizeof(aBits));
    std::memcpy(&bBits, &b, sizeof(bBits));
    return aBits == bBits;
}

bool sameVec3(const glm::vec3& a, const glm::vec3& b) {
    return sameFloat(a.x, b.x) && sameFloat(a.y, b.y) &&
           sameFloat(a.z, b.z);
}

CheckpointData makeCheckpoint() {
    CheckpointData data;
    data.completedSamples = 17;
    data.seed = 0x0123456789abcdefull;
    data.samplerVersion = 3;
    data.config.width = 2;
    data.config.height = 2;
    data.config.maxDepth = 8;
    data.config.antialias = true;
    data.config.nextEventEstimation = true;
    data.config.useBvh = true;
    data.config.compact = false;
    data.config.materialSort = true;
    data.config.wavefront = false;
    data.config.russianRoulette = true;
    // A start depth beyond maxDepth is valid and remains part of the exact
    // continuation configuration.
    data.config.rrStart = 11;
    data.config.motion = false;

    data.camera.position = glm::vec3(1.0f, 2.0f, 3.0f);
    data.camera.lookAt = glm::vec3(0.0f, 0.5f, -1.0f);
    data.camera.view = glm::vec3(-1.0f, -1.5f, -4.0f);
    data.camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
    data.camera.right = glm::vec3(1.0f, 0.0f, 0.0f);
    data.camera.fov = glm::vec2(0.8f, 0.6f);
    data.camera.pixelLength = glm::vec2(0.004f, 0.005f);
    data.camera.aperture = 0.125f;
    data.camera.focusDistance = 4.25f;
    data.camera.shutterOpen = 0.25f;
    data.camera.shutterClose = 0.75f;

    data.compatibility.cudaRuntimeVersion = 12040;
    data.compatibility.cudaDriverVersion = 12050;
    data.compatibility.computeCapabilityMajor = 8;
    data.compatibility.computeCapabilityMinor = 6;
    data.compatibility.deviceIdentifier = "unit-test-device";
    data.compatibility.sourceRevision =
        "8695a5af18ae6fc130d1d38713be82bbbff7f38f";
    data.compatibility.buildIdentifier = "checkpoint-host-test";

    data.fingerprints.sceneFile = 0x11;
    data.fingerprints.assets = 0x22;
    data.fingerprints.materials = 0x33;
    data.fingerprints.geometry = 0x44;
    data.fingerprints.renderConfig = 0x55;

    data.accumulation = {
        glm::vec3(0.0f, -0.0f, 1.0f),
        glm::vec3(1.25f, 2.5f, 3.75f),
        glm::vec3(1000.0f, 0.0001f, 42.0f),
        glm::vec3(7.0f, 8.0f, 9.0f)};

    checkpoint::BvhNodeRecord root;
    root.boundsMin = glm::vec3(-2.0f);
    root.boundsMax = glm::vec3(2.0f);
    root.left = 1;
    root.right = 2;
    root.first = 0;
    root.count = 0;
    checkpoint::BvhNodeRecord left;
    left.boundsMin = glm::vec3(-2.0f, -1.0f, -1.0f);
    left.boundsMax = glm::vec3(0.0f, 1.0f, 1.0f);
    left.first = 0;
    left.count = 1;
    checkpoint::BvhNodeRecord right;
    right.boundsMin = glm::vec3(0.0f, -1.0f, -1.0f);
    right.boundsMax = glm::vec3(2.0f, 1.0f, 1.0f);
    right.first = 1;
    right.count = 1;
    data.bvhNodes = {root, left, right};
    data.bvhPrimitiveIndices = {1, 0};
    return data;
}

checkpoint::ExpectedState expectedFrom(const CheckpointData& data) {
    checkpoint::ExpectedState expected;
    expected.seed = data.seed;
    expected.samplerVersion = data.samplerVersion;
    expected.config = data.config;
    expected.camera = data.camera;
    expected.compatibility = data.compatibility;
    expected.fingerprints = data.fingerprints;
    return expected;
}

void requireEquivalent(const CheckpointData& a, const CheckpointData& b) {
    require(a.completedSamples == b.completedSamples, "sample count changed");
    require(a.seed == b.seed, "seed changed");
    require(a.samplerVersion == b.samplerVersion, "sampler version changed");
    require(a.config.width == b.config.width &&
                a.config.height == b.config.height &&
                a.config.maxDepth == b.config.maxDepth &&
                a.config.antialias == b.config.antialias &&
                a.config.nextEventEstimation == b.config.nextEventEstimation &&
                a.config.useBvh == b.config.useBvh &&
                a.config.compact == b.config.compact &&
                a.config.materialSort == b.config.materialSort &&
                a.config.wavefront == b.config.wavefront &&
                a.config.russianRoulette == b.config.russianRoulette &&
                a.config.rrStart == b.config.rrStart &&
                a.config.motion == b.config.motion,
            "render config changed");
    require(sameVec3(a.camera.position, b.camera.position) &&
                sameVec3(a.camera.lookAt, b.camera.lookAt) &&
                sameVec3(a.camera.view, b.camera.view) &&
                sameVec3(a.camera.up, b.camera.up) &&
                sameVec3(a.camera.right, b.camera.right) &&
                sameFloat(a.camera.fov.x, b.camera.fov.x) &&
                sameFloat(a.camera.fov.y, b.camera.fov.y) &&
                sameFloat(a.camera.pixelLength.x, b.camera.pixelLength.x) &&
                sameFloat(a.camera.pixelLength.y, b.camera.pixelLength.y) &&
                sameFloat(a.camera.aperture, b.camera.aperture) &&
                sameFloat(a.camera.focusDistance, b.camera.focusDistance) &&
                sameFloat(a.camera.shutterOpen, b.camera.shutterOpen) &&
                sameFloat(a.camera.shutterClose, b.camera.shutterClose),
            "camera changed");
    require(a.compatibility.deviceIdentifier ==
                    b.compatibility.deviceIdentifier &&
                a.compatibility.sourceRevision ==
                    b.compatibility.sourceRevision &&
                a.compatibility.buildIdentifier ==
                    b.compatibility.buildIdentifier &&
                a.compatibility.cudaRuntimeVersion ==
                    b.compatibility.cudaRuntimeVersion &&
                a.compatibility.cudaDriverVersion ==
                    b.compatibility.cudaDriverVersion &&
                a.compatibility.computeCapabilityMajor ==
                    b.compatibility.computeCapabilityMajor &&
                a.compatibility.computeCapabilityMinor ==
                    b.compatibility.computeCapabilityMinor,
            "compatibility metadata changed");
    require(a.fingerprints.sceneFile == b.fingerprints.sceneFile &&
                a.fingerprints.assets == b.fingerprints.assets &&
                a.fingerprints.materials == b.fingerprints.materials &&
                a.fingerprints.geometry == b.fingerprints.geometry &&
                a.fingerprints.renderConfig == b.fingerprints.renderConfig,
            "fingerprints changed");
    require(a.accumulation.size() == b.accumulation.size(),
            "accumulation size changed");
    for (std::size_t i = 0; i < a.accumulation.size(); ++i) {
        require(sameVec3(a.accumulation[i], b.accumulation[i]),
                "accumulation value changed");
    }
    require(a.bvhNodes.size() == b.bvhNodes.size(), "BVH size changed");
    for (std::size_t i = 0; i < a.bvhNodes.size(); ++i) {
        require(sameVec3(a.bvhNodes[i].boundsMin, b.bvhNodes[i].boundsMin) &&
                    sameVec3(a.bvhNodes[i].boundsMax, b.bvhNodes[i].boundsMax) &&
                    a.bvhNodes[i].left == b.bvhNodes[i].left &&
                    a.bvhNodes[i].right == b.bvhNodes[i].right &&
                    a.bvhNodes[i].first == b.bvhNodes[i].first &&
                    a.bvhNodes[i].count == b.bvhNodes[i].count,
                "BVH node changed");
    }
    require(a.bvhPrimitiveIndices == b.bvhPrimitiveIndices,
            "BVH indices changed");
}

std::vector<std::uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(stream.is_open(), "could not open test file");
    const std::streamsize size = stream.tellg();
    require(size >= 0, "could not determine test file size");
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    require(stream.good() || stream.eof(), "could not read test file");
    return bytes;
}

void writeBytes(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.is_open(), "could not create test file");
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    stream.close();
    require(!stream.fail(), "could not write test file");
}

void writeText(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.is_open(), "could not create scene fixture");
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.close();
    require(!stream.fail(), "could not write scene fixture");
}

void putU32(std::vector<std::uint8_t>& bytes,
            std::size_t offset,
            std::uint32_t value) {
    require(offset + 4 <= bytes.size(), "test mutation offset out of range");
    for (unsigned i = 0; i < 4; ++i) {
        bytes[offset + i] =
            static_cast<std::uint8_t>((value >> (i * 8u)) & 0xffu);
    }
}

std::uint32_t getU32(const std::vector<std::uint8_t>& bytes,
                     std::size_t offset) {
    require(offset + 4 <= bytes.size(), "test read offset out of range");
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(bytes[offset + i]) << (i * 8u);
    }
    return value;
}

void putU64(std::vector<std::uint8_t>& bytes,
            std::size_t offset,
            std::uint64_t value) {
    require(offset + 8 <= bytes.size(), "test mutation offset out of range");
    for (unsigned i = 0; i < 8; ++i) {
        bytes[offset + i] =
            static_cast<std::uint8_t>((value >> (i * 8u)) & 0xffu);
    }
}

std::uint64_t getU64(const std::vector<std::uint8_t>& bytes,
                     std::size_t offset) {
    require(offset + 8 <= bytes.size(), "test read offset out of range");
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (i * 8u);
    }
    return value;
}

void refreshChecksum(std::vector<std::uint8_t>& bytes) {
    require(bytes.size() >= 8, "test file too short for checksum");
    const std::size_t checksumOffset = bytes.size() - 8;
    const std::uint64_t checksum = checkpoint::fnv1aUpdate(
        checkpoint::kFnv1aOffsetBasis, bytes.data(), checksumOffset);
    putU64(bytes, checksumOffset, checksum);
}

class DirectoryGuard {
public:
    explicit DirectoryGuard(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~DirectoryGuard() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

private:
    std::filesystem::path path_;
};

void testRoundTrip(const std::filesystem::path& directory) {
    const CheckpointData original = makeCheckpoint();
    const std::filesystem::path first = directory / "round-trip.ckpt";
    const std::filesystem::path second = directory / "round-trip-2.ckpt";
    checkpoint::Status status = checkpoint::saveAtomic(first, original);
    require(status.ok(), "save failed: " + status.message);

    checkpoint::CheckpointMetadata metadata;
    status = checkpoint::inspect(first, metadata);
    require(status.ok(), "metadata inspection failed: " + status.message);
    require(metadata.completedSamples == original.completedSamples &&
                metadata.seed == original.seed &&
                metadata.samplerVersion == original.samplerVersion &&
                metadata.config.width == original.config.width &&
                metadata.config.height == original.config.height &&
                metadata.config.maxDepth == original.config.maxDepth &&
                metadata.config.antialias == original.config.antialias &&
                metadata.config.nextEventEstimation ==
                    original.config.nextEventEstimation &&
                metadata.config.useBvh == original.config.useBvh &&
                metadata.config.compact == original.config.compact &&
                metadata.config.materialSort ==
                    original.config.materialSort &&
                metadata.config.wavefront == original.config.wavefront &&
                metadata.config.russianRoulette ==
                    original.config.russianRoulette &&
                metadata.config.rrStart == original.config.rrStart &&
                metadata.config.motion == original.config.motion &&
                sameFloat(metadata.camera.shutterOpen,
                          original.camera.shutterOpen) &&
                sameFloat(metadata.camera.shutterClose,
                          original.camera.shutterClose) &&
                metadata.compatibility.buildIdentifier ==
                    original.compatibility.buildIdentifier &&
                metadata.fingerprints.geometry ==
                    original.fingerprints.geometry &&
                metadata.accumulationCount == original.accumulation.size() &&
                metadata.bvhNodeCount == original.bvhNodes.size() &&
                metadata.bvhPrimitiveIndexCount ==
                    original.bvhPrimitiveIndices.size(),
            "metadata inspection changed or omitted a field");

    CheckpointData loaded;
    const checkpoint::ExpectedState expected = expectedFrom(original);
    status = checkpoint::load(first, loaded, &expected);
    require(status.ok(), "load failed: " + status.message);
    requireEquivalent(original, loaded);

    status = checkpoint::saveAtomic(second, loaded);
    require(status.ok(), "second save failed: " + status.message);
    require(readBytes(first) == readBytes(second),
            "canonical round-trip was not byte-identical");

    // Replacing an existing path must leave the newly completed boundary.
    loaded.completedSamples = 18;
    status = checkpoint::saveAtomic(first, loaded);
    require(status.ok(), "atomic replacement failed: " + status.message);
    CheckpointData replaced;
    status = checkpoint::load(first, replaced);
    require(status.ok() && replaced.completedSamples == 18,
            "atomic replacement did not publish the new checkpoint");
}

void testFormatLayoutAndSamplerIds(const std::filesystem::path& directory) {
    const std::filesystem::path haltonPath = directory / "format-v3-halton.ckpt";
    const CheckpointData halton = makeCheckpoint();
    checkpoint::Status status = checkpoint::saveAtomic(haltonPath, halton);
    require(status.ok(), "v3 Halton layout fixture could not be saved");
    const std::vector<std::uint8_t> bytes = readBytes(haltonPath);
    require(bytes.size() > 160, "v3 checkpoint is shorter than fixed metadata");
    require(getU32(bytes, 8) == 3u, "checkpoint header is not format v3");
    require(getU64(bytes, 16) == bytes.size() - 24u - 8u,
            "v3 payload byte count is inconsistent with file length");
    require(getU32(bytes, 40) == 3u,
            "Halton sampler ID is not at absolute offset 40");
    require(bytes[67] == 0u,
            "motion boolean is not at absolute offset 67");
    require(getU32(bytes, 152) == [] {
                std::uint32_t bits = 0;
                const float value = 0.25f;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }(),
            "shutter-open float is not at absolute offset 152");
    require(getU32(bytes, 156) == [] {
                std::uint32_t bits = 0;
                const float value = 0.75f;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }(),
            "shutter-close float is not at absolute offset 156");

    CheckpointData counter = makeCheckpoint();
    counter.samplerVersion = 2;
    counter.config.motion = true;
    counter.camera.shutterOpen = 0.5f;
    counter.camera.shutterClose = 0.5f;
    const std::filesystem::path counterPath =
        directory / "format-v3-counter.ckpt";
    status = checkpoint::saveAtomic(counterPath, counter);
    require(status.ok(), "v3 counter checkpoint could not be saved");

    checkpoint::CheckpointMetadata metadata;
    status = checkpoint::inspect(counterPath, metadata);
    require(status.ok(), "v3 counter checkpoint inspection failed");
    require(metadata.samplerVersion == 2u && metadata.config.motion &&
                sameFloat(metadata.camera.shutterOpen, 0.5f) &&
                sameFloat(metadata.camera.shutterClose, 0.5f),
            "inspect did not expose sampler/motion/frozen-shutter inheritance data");

    CheckpointData loaded;
    const checkpoint::ExpectedState expected = expectedFrom(counter);
    status = checkpoint::load(counterPath, loaded, &expected);
    require(status.ok(), "v3 counter checkpoint load failed");
    requireEquivalent(counter, loaded);
    require(getU32(readBytes(counterPath), 40) == 2u,
            "counter sampler ID was not serialized independently of format v3");
}

void testSamplerIdValidation(const std::filesystem::path& directory) {
    CheckpointData invalid = makeCheckpoint();
    invalid.samplerVersion = 4;
    checkpoint::Status status = checkpoint::validate(invalid);
    require(status.code == checkpoint::Error::Malformed,
            "validation accepted an unknown sampler ID");
    status = checkpoint::saveAtomic(directory / "unknown-sampler-save.ckpt",
                                    invalid);
    require(status.code == checkpoint::Error::Malformed,
            "save accepted an unknown sampler ID");

    const std::filesystem::path path =
        directory / "unknown-sampler-encoded.ckpt";
    status = checkpoint::saveAtomic(path, makeCheckpoint());
    require(status.ok(), "unknown sampler load fixture could not be saved");
    std::vector<std::uint8_t> bytes = readBytes(path);
    putU32(bytes, 40, 99u);
    refreshChecksum(bytes);
    writeBytes(path, bytes);

    checkpoint::CheckpointMetadata metadata;
    status = checkpoint::inspect(path, metadata);
    require(status.code == checkpoint::Error::Malformed,
            "inspection accepted a checksummed unknown sampler ID");
    CheckpointData loaded;
    status = checkpoint::load(path, loaded);
    require(status.code == checkpoint::Error::Malformed,
            "load accepted a checksummed unknown sampler ID");
}

void testCorruption(const std::filesystem::path& directory) {
    const std::filesystem::path path = directory / "corrupt.ckpt";
    CheckpointData data = makeCheckpoint();
    checkpoint::Status status = checkpoint::saveAtomic(path, data);
    require(status.ok(), "corruption test setup failed");
    std::vector<std::uint8_t> bytes = readBytes(path);
    require(bytes.size() > 80, "checkpoint unexpectedly small");
    bytes[70] ^= 0x40u;
    writeBytes(path, bytes);

    CheckpointData sentinel;
    sentinel.completedSamples = 999;
    sentinel.accumulation.push_back(glm::vec3(9.0f));
    status = checkpoint::load(path, sentinel);
    require(status.code == checkpoint::Error::ChecksumMismatch,
            "corruption was not reported as checksum mismatch");
    require(sentinel.completedSamples == 999 &&
                sentinel.accumulation.size() == 1,
            "failed load modified its output object");

    checkpoint::CheckpointMetadata metadata;
    metadata.completedSamples = 777;
    status = checkpoint::inspect(path, metadata);
    require(status.code == checkpoint::Error::ChecksumMismatch,
            "metadata inspection did not checksum the entire file");
    require(metadata.completedSamples == 777,
            "failed metadata inspection modified its output object");
}

void testContinuationMismatch(const std::filesystem::path& directory) {
    const std::filesystem::path path = directory / "mismatch.ckpt";
    const CheckpointData data = makeCheckpoint();
    checkpoint::Status status = checkpoint::saveAtomic(path, data);
    require(status.ok(), "mismatch test setup failed");

    checkpoint::ExpectedState expected = expectedFrom(data);
    expected.config.materialSort = !expected.config.materialSort;
    CheckpointData loaded;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "config mismatch was not rejected");

    expected = expectedFrom(data);
    expected.config.wavefront = !expected.config.wavefront;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "wavefront mismatch was not rejected");

    expected = expectedFrom(data);
    expected.config.russianRoulette = !expected.config.russianRoulette;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "Russian-roulette mismatch was not rejected");

    expected = expectedFrom(data);
    ++expected.config.rrStart;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "Russian-roulette start-depth mismatch was not rejected");

    expected = expectedFrom(data);
    expected.config.motion = !expected.config.motion;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "motion mismatch was not rejected");

    expected = expectedFrom(data);
    expected.camera.shutterOpen = 0.125f;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "shutter-open mismatch was not rejected");

    expected = expectedFrom(data);
    expected.camera.shutterClose = 1.0f;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "shutter-close mismatch was not rejected");

    expected = expectedFrom(data);
    expected.samplerVersion = 2;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "sampler-ID mismatch was not rejected");

    expected = expectedFrom(data);
    expected.fingerprints.assets ^= 1u;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "asset fingerprint mismatch was not rejected");

    expected = expectedFrom(data);
    expected.fingerprints.geometry ^= 1u;
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "geometry fingerprint mismatch was not rejected");

    expected = expectedFrom(data);
    expected.compatibility.buildIdentifier += "-different";
    status = checkpoint::load(path, loaded, &expected);
    require(status.code == checkpoint::Error::ContinuationMismatch,
            "build compatibility mismatch was not rejected");
}

void testSizeValidation(const std::filesystem::path& directory) {
    const std::filesystem::path path = directory / "size.ckpt";
    CheckpointData data = makeCheckpoint();
    checkpoint::Status status = checkpoint::saveAtomic(path, data);
    require(status.ok(), "size test setup failed");

    checkpoint::Limits tight;
    tight.maxPixels = 3;
    CheckpointData loaded;
    status = checkpoint::load(path, loaded, nullptr, tight);
    require(status.code == checkpoint::Error::LimitExceeded,
            "load limit did not reject pixel count before allocation");
    checkpoint::CheckpointMetadata metadata;
    status = checkpoint::inspect(path, metadata, tight);
    require(status.code == checkpoint::Error::LimitExceeded,
            "metadata inspection did not enforce pixel-count limit");

    std::vector<std::uint8_t> bytes = readBytes(path);
    // Header is 24 bytes; width is 20 bytes into the payload.
    putU32(bytes, 24 + 20, std::numeric_limits<std::uint32_t>::max());
    refreshChecksum(bytes);
    writeBytes(path, bytes);
    status = checkpoint::load(path, loaded);
    require(status.code == checkpoint::Error::LimitExceeded,
            "checksummed malicious resolution was not rejected");

    data.accumulation.pop_back();
    status = checkpoint::saveAtomic(directory / "wrong-count.ckpt", data);
    require(status.code == checkpoint::Error::Malformed,
            "save accepted accumulation/resolution size mismatch");
}

void testNonFiniteValidation(const std::filesystem::path& directory) {
    const std::filesystem::path path = directory / "nan.ckpt";
    CheckpointData data = makeCheckpoint();
    data.accumulation[1].y = std::numeric_limits<float>::quiet_NaN();
    checkpoint::Status status = checkpoint::saveAtomic(path, data);
    require(status.code == checkpoint::Error::NonFinite,
            "save accepted NaN accumulation");

    data = makeCheckpoint();
    status = checkpoint::saveAtomic(path, data);
    require(status.ok(), "NaN load test setup failed");
    std::vector<std::uint8_t> bytes = readBytes(path);
    // Camera position.x starts after 20 bytes of sample metadata and the
    // 24-byte v3 RenderConfig in the payload.
    putU32(bytes, 24 + 20 + 24, 0x7fc00000u);
    refreshChecksum(bytes);
    writeBytes(path, bytes);
    CheckpointData loaded;
    status = checkpoint::load(path, loaded);
    require(status.code == checkpoint::Error::NonFinite,
            "checksummed NaN camera value was not rejected");
    checkpoint::CheckpointMetadata metadata;
    status = checkpoint::inspect(path, metadata);
    require(status.code == checkpoint::Error::NonFinite,
            "metadata inspection accepted a checksummed NaN camera value");
}

void testProjectionValidation(const std::filesystem::path& directory) {
    const std::filesystem::path scenePath = directory / "zero-fov-scene.json";
    writeText(
        scenePath,
        R"({"Materials":{"mat":{"TYPE":"Diffuse","RGB":[1,1,1]}},)"
        R"("Objects":[{"TYPE":"sphere","MATERIAL":"mat","TRANS":[0,0,0],)"
        R"("ROTAT":[0,0,0],"SCALE":[1,1,1]}],)"
        R"("Camera":{"RES":[2,2],"FOVY":0,"ITERATIONS":1,"DEPTH":1,)"
        R"("FILE":"unused","EYE":[0,0,1],"LOOKAT":[0,0,0],"UP":[0,1,0]}})"
    );
    const Scene scene(scenePath.string());
    const Camera& accepted = scene.state.camera;
    require(sameFloat(accepted.fov.x, 0.0f),
            "zero-FOV scene did not produce zero horizontal FOV");
    require(sameFloat(accepted.fov.y, 0.0f),
            "zero-FOV scene did not preserve zero vertical FOV");
    require(sameFloat(accepted.pixelLength.x, 0.0f),
            "zero-FOV scene did not produce zero horizontal pixel length");
    require(sameFloat(accepted.pixelLength.y, 0.0f),
            "zero-FOV scene did not produce zero vertical pixel length");

    CheckpointData zeroProjection = makeCheckpoint();
    zeroProjection.camera.position = accepted.position;
    zeroProjection.camera.lookAt = accepted.lookAt;
    zeroProjection.camera.view = accepted.view;
    zeroProjection.camera.up = accepted.up;
    zeroProjection.camera.right = accepted.right;
    zeroProjection.camera.fov = accepted.fov;
    zeroProjection.camera.pixelLength = accepted.pixelLength;
    zeroProjection.camera.aperture = accepted.aperture;
    zeroProjection.camera.focusDistance = accepted.focusDistance;
    zeroProjection.camera.shutterOpen = accepted.shutterOpen;
    zeroProjection.camera.shutterClose = accepted.shutterClose;

    const std::filesystem::path checkpointPath =
        directory / "zero-fov-round-trip.ckpt";
    checkpoint::Status status =
        checkpoint::saveAtomic(checkpointPath, zeroProjection);
    require(status.ok(), "zero-FOV checkpoint save failed: " + status.message);
    CheckpointData loaded;
    const checkpoint::ExpectedState expected = expectedFrom(zeroProjection);
    status = checkpoint::load(checkpointPath, loaded, &expected);
    require(status.ok(), "zero-FOV checkpoint load failed: " + status.message);
    requireEquivalent(zeroProjection, loaded);

    const char* componentNames[] = {
        "fov.x", "fov.y", "pixelLength.x", "pixelLength.y"};
    const auto setComponent = [](checkpoint::CameraState& camera,
                                 std::size_t component,
                                 float value) {
        switch (component) {
        case 0: camera.fov.x = value; break;
        case 1: camera.fov.y = value; break;
        case 2: camera.pixelLength.x = value; break;
        case 3: camera.pixelLength.y = value; break;
        default: throw std::logic_error("unknown projection component");
        }
    };

    for (std::size_t component = 0; component < 4; ++component) {
        CheckpointData invalid = makeCheckpoint();
        setComponent(invalid.camera, component, -0.25f);
        status = checkpoint::validate(invalid);
        require(status.code == checkpoint::Error::Malformed,
                std::string("negative ") + componentNames[component] +
                    " was accepted");

        invalid = makeCheckpoint();
        setComponent(invalid.camera, component,
                     std::numeric_limits<float>::quiet_NaN());
        status = checkpoint::validate(invalid);
        require(status.code == checkpoint::Error::NonFinite,
                std::string("NaN ") + componentNames[component] +
                    " was accepted");

        invalid = makeCheckpoint();
        setComponent(invalid.camera, component,
                     std::numeric_limits<float>::infinity());
        status = checkpoint::validate(invalid);
        require(status.code == checkpoint::Error::NonFinite,
                std::string("infinite ") + componentNames[component] +
                    " was accepted");
    }
}

void testShutterValidation(const std::filesystem::path& directory) {
    CheckpointData data = makeCheckpoint();
    data.camera.shutterOpen = 0.6f;
    data.camera.shutterClose = 0.6f;
    checkpoint::Status status = checkpoint::validate(data);
    require(status.ok(), "frozen shutter was rejected");

    data.camera.shutterOpen = 0.8f;
    data.camera.shutterClose = 0.7f;
    status = checkpoint::validate(data);
    require(status.code == checkpoint::Error::Malformed,
            "reversed shutter interval was accepted");

    data = makeCheckpoint();
    data.camera.shutterOpen = -0.01f;
    status = checkpoint::validate(data);
    require(status.code == checkpoint::Error::Malformed,
            "negative shutter-open value was accepted");

    data = makeCheckpoint();
    data.camera.shutterClose = 1.01f;
    status = checkpoint::saveAtomic(directory / "shutter-out-of-range.ckpt",
                                    data);
    require(status.code == checkpoint::Error::Malformed,
            "save accepted shutter-close above one");

    data = makeCheckpoint();
    data.camera.shutterOpen = std::numeric_limits<float>::quiet_NaN();
    status = checkpoint::validate(data);
    require(status.code == checkpoint::Error::NonFinite,
            "NaN shutter-open value was accepted");

    data = makeCheckpoint();
    data.camera.shutterClose = std::numeric_limits<float>::infinity();
    status = checkpoint::saveAtomic(directory / "shutter-infinite.ckpt", data);
    require(status.code == checkpoint::Error::NonFinite,
            "save accepted infinite shutter-close value");

    struct Mutation {
        const char* name;
        std::size_t offset;
        std::uint32_t bits;
        checkpoint::Error expectedError;
    };
    std::uint32_t negativeBits = 0;
    const float negative = -0.25f;
    std::memcpy(&negativeBits, &negative, sizeof(negativeBits));
    std::uint32_t reversedBits = 0;
    const float reversed = 0.9f;
    std::memcpy(&reversedBits, &reversed, sizeof(reversedBits));
    std::uint32_t highBits = 0;
    const float high = 1.25f;
    std::memcpy(&highBits, &high, sizeof(highBits));
    const Mutation mutations[] = {
        {"negative", 152, negativeBits, checkpoint::Error::Malformed},
        {"reversed", 152, reversedBits, checkpoint::Error::Malformed},
        {"above-one", 156, highBits, checkpoint::Error::Malformed},
        {"nonfinite", 156, 0x7fc00000u, checkpoint::Error::NonFinite},
    };
    for (const Mutation& mutation : mutations) {
        const std::filesystem::path path =
            directory / (std::string("shutter-corrupt-") + mutation.name +
                         ".ckpt");
        status = checkpoint::saveAtomic(path, makeCheckpoint());
        require(status.ok(), "shutter corruption fixture could not be saved");
        std::vector<std::uint8_t> bytes = readBytes(path);
        putU32(bytes, mutation.offset, mutation.bits);
        refreshChecksum(bytes);
        writeBytes(path, bytes);

        checkpoint::CheckpointMetadata metadata;
        status = checkpoint::inspect(path, metadata);
        require(status.code == mutation.expectedError,
                std::string("inspection accepted checksummed ") +
                    mutation.name + " shutter metadata");
        CheckpointData loaded;
        status = checkpoint::load(path, loaded);
        require(status.code == mutation.expectedError,
                std::string("load accepted checksummed ") + mutation.name +
                    " shutter metadata");
    }
}

void testRussianRouletteBounds() {
    CheckpointData data = makeCheckpoint();
    data.config.russianRoulette = false;

    data.config.rrStart = 0;
    checkpoint::Status status = checkpoint::validate(data);
    require(status.code == checkpoint::Error::Malformed,
            "zero Russian-roulette start depth was accepted while RR was off");

    data.config.rrStart = 1;
    status = checkpoint::validate(data);
    require(status.ok(), "minimum Russian-roulette start depth was rejected");

    data.config.rrStart = 4096;
    status = checkpoint::validate(data);
    require(status.ok(),
            "maximum Russian-roulette start depth was rejected when beyond maxDepth");

    data.config.rrStart = 4097;
    status = checkpoint::validate(data);
    require(status.code == checkpoint::Error::LimitExceeded,
            "Russian-roulette start depth above 4096 was accepted while RR was off");
}

void testConfigEncodingRejection(const std::filesystem::path& directory) {
    const CheckpointData data = makeCheckpoint();

    // The scheduling fields follow the 17-byte v1 RenderConfig prefix, and
    // v3 appends motion after rrStart.
    const std::size_t configOffset = 24 + 20;
    const std::size_t wavefrontOffset = configOffset + 17;
    const std::size_t russianRouletteOffset = configOffset + 18;
    const std::size_t motionOffset = configOffset + 23;
    const std::size_t boolOffsets[] = {
        wavefrontOffset, russianRouletteOffset, motionOffset};
    for (std::size_t i = 0; i < 3; ++i) {
        const std::filesystem::path path =
            directory / ("noncanonical-config-" + std::to_string(i) + ".ckpt");
        checkpoint::Status status = checkpoint::saveAtomic(path, data);
        require(status.ok(), "non-canonical boolean test setup failed");
        std::vector<std::uint8_t> bytes = readBytes(path);
        require(boolOffsets[i] < bytes.size(),
                "non-canonical boolean mutation offset is out of range");
        bytes[boolOffsets[i]] = 2u;
        refreshChecksum(bytes);
        writeBytes(path, bytes);

        checkpoint::CheckpointMetadata metadata;
        status = checkpoint::inspect(path, metadata);
        require(status.code == checkpoint::Error::Malformed,
                "inspection accepted a non-canonical config boolean");
        CheckpointData loaded;
        status = checkpoint::load(path, loaded);
        require(status.code == checkpoint::Error::Malformed,
                "load accepted a non-canonical config boolean");
    }
}

void testSchedulingConflictRejection(const std::filesystem::path& directory) {
    CheckpointData invalid = makeCheckpoint();
    invalid.config.wavefront = true;
    require(invalid.config.materialSort,
            "scheduling-conflict fixture must enable material sorting");

    checkpoint::Status status = checkpoint::validate(invalid);
    require(status.code == checkpoint::Error::Malformed,
            "validation accepted wavefront scheduling with material sorting");
    const std::filesystem::path rejected = directory / "conflict-save.ckpt";
    status = checkpoint::saveAtomic(rejected, invalid);
    require(status.code == checkpoint::Error::Malformed,
            "save accepted wavefront scheduling with material sorting");

    const std::filesystem::path encoded = directory / "conflict-encoded.ckpt";
    CheckpointData valid = makeCheckpoint();
    status = checkpoint::saveAtomic(encoded, valid);
    require(status.ok(), "scheduling-conflict load fixture could not be saved");
    std::vector<std::uint8_t> bytes = readBytes(encoded);
    const std::size_t wavefrontOffset = 24 + 20 + 17;
    require(bytes[wavefrontOffset] == 0u,
            "scheduling-conflict fixture unexpectedly enables wavefront mode");
    bytes[wavefrontOffset] = 1u;
    refreshChecksum(bytes);
    writeBytes(encoded, bytes);

    checkpoint::CheckpointMetadata metadata;
    status = checkpoint::inspect(encoded, metadata);
    require(status.code == checkpoint::Error::Malformed,
            "inspection accepted wavefront scheduling with material sorting");
    CheckpointData loaded;
    status = checkpoint::load(encoded, loaded);
    require(status.code == checkpoint::Error::Malformed,
            "load accepted wavefront scheduling with material sorting");
}

void testOldVersionRejection(const std::filesystem::path& directory) {
    for (const std::uint32_t version : {1u, 2u}) {
        const std::filesystem::path path =
            directory / ("old-version-" + std::to_string(version) + ".ckpt");
        const CheckpointData data = makeCheckpoint();
        checkpoint::Status status = checkpoint::saveAtomic(path, data);
        require(status.ok(), "old-version rejection test setup failed");

        std::vector<std::uint8_t> bytes = readBytes(path);
        putU32(bytes, 8, version);
        refreshChecksum(bytes);
        writeBytes(path, bytes);

        checkpoint::CheckpointMetadata metadata;
        metadata.completedSamples = 999;
        status = checkpoint::inspect(path, metadata);
        require(status.code == checkpoint::Error::UnsupportedVersion,
                "inspection did not strictly reject old checkpoint format");
        require(metadata.completedSamples == 999,
                "failed old-format inspection modified its output object");

        CheckpointData loaded;
        loaded.completedSamples = 777;
        status = checkpoint::load(path, loaded);
        require(status.code == checkpoint::Error::UnsupportedVersion,
                "load did not strictly reject old checkpoint format");
        require(loaded.completedSamples == 777,
                "failed old-format load modified its output object");
    }
}

void testFingerprinting(const std::filesystem::path& directory) {
    const char hello[] = "hello";
    require(checkpoint::fnv1aUpdate(checkpoint::kFnv1aOffsetBasis, hello, 5) ==
                0xa430d84680aabd0bull,
            "FNV-1a implementation failed known-answer test");

    checkpoint::FingerprintBuilder a;
    checkpoint::FingerprintBuilder b;
    a.addString("tag").addU32(9).addFloat(0.5f).addBool(true);
    b.addString("tag").addU32(9).addFloat(0.5f).addBool(true);
    require(a.value() == b.value(), "canonical fingerprint is unstable");
    b.addU32(1);
    require(a.value() != b.value(), "fingerprint ignored appended field");

    const std::filesystem::path path = directory / "fingerprint.bin";
    writeBytes(path, std::vector<std::uint8_t>(hello, hello + 5));
    std::uint64_t fileFingerprint = 0;
    const checkpoint::Status status =
        checkpoint::fingerprintFile(path, fileFingerprint);
    require(status.ok() && fileFingerprint == 0xa430d84680aabd0bull,
            "streamed file fingerprint is incorrect");
}

void testBufferedBoundaries(const std::filesystem::path& directory) {
    CheckpointData data = makeCheckpoint();
    // Force a single explicitly serialized field to cross several 64 KiB I/O
    // buffer boundaries.  This also exercises payload/checksum separation when
    // a field does not align with a buffer boundary.
    data.compatibility.buildIdentifier.assign(150000, 'b');
    const std::filesystem::path path = directory / "large-string.ckpt";
    checkpoint::Status status = checkpoint::saveAtomic(path, data);
    require(status.ok(), "large-field save failed: " + status.message);
    CheckpointData loaded;
    status = checkpoint::load(path, loaded);
    require(status.ok(), "large-field load failed: " + status.message);
    require(loaded.compatibility.buildIdentifier ==
                data.compatibility.buildIdentifier,
            "buffer-crossing field changed during round-trip");
}

void expectMalformedBvh(const CheckpointData& data, const std::string& caseName) {
    const checkpoint::Status status = checkpoint::validate(data);
    require(status.code == checkpoint::Error::Malformed,
            "malformed BVH case was accepted: " + caseName +
                (status.message.empty() ? std::string() : " (" + status.message + ")"));
}

void testBvhValidation() {
    CheckpointData data = makeCheckpoint();
    data.bvhNodes[0].left = 0;
    expectMalformedBvh(data, "cycle/self-child");

    data = makeCheckpoint();
    data.bvhNodes[0].right = data.bvhNodes[0].left;
    expectMalformedBvh(data, "shared sibling");

    data = makeCheckpoint();
    data.bvhNodes[0].right = -1;
    expectMalformedBvh(data, "missing internal child");

    data = makeCheckpoint();
    data.bvhNodes[1].left = 2;
    expectMalformedBvh(data, "leaf with child");

    data = makeCheckpoint();
    data.bvhNodes.push_back(data.bvhNodes[1]);
    expectMalformedBvh(data, "unreachable node");

    data = makeCheckpoint();
    data.bvhNodes[2].first = 0;
    expectMalformedBvh(data, "overlapping leaf ranges");

    data = makeCheckpoint();
    data.bvhPrimitiveIndices.push_back(2);
    expectMalformedBvh(data, "gap in leaf ranges");

    data = makeCheckpoint();
    data.bvhPrimitiveIndices = {0, 0};
    expectMalformedBvh(data, "duplicate primitive permutation entry");

    data = makeCheckpoint();
    data.bvhPrimitiveIndices = {0, 2};
    expectMalformedBvh(data, "out-of-range primitive permutation entry");

    data = makeCheckpoint();
    data.bvhNodes.clear();
    expectMalformedBvh(data, "empty nodes with nonempty indices");

    // A binary comb with an internal node at depth 60 would require the GPU
    // traversal to visit children at depth 61, beyond geometry.cuh's contract.
    data = makeCheckpoint();
    constexpr std::int32_t internalCount = 61;
    constexpr std::int32_t sideLeafStart = internalCount;
    constexpr std::int32_t finalLeaf = internalCount * 2;
    data.bvhNodes.assign(static_cast<std::size_t>(finalLeaf + 1), {});
    data.bvhPrimitiveIndices.resize(static_cast<std::size_t>(internalCount + 1));
    for (std::int32_t i = 0; i <= internalCount; ++i) {
        data.bvhPrimitiveIndices[static_cast<std::size_t>(i)] = i;
    }
    for (std::int32_t depth = 0; depth < internalCount; ++depth) {
        checkpoint::BvhNodeRecord& internal =
            data.bvhNodes[static_cast<std::size_t>(depth)];
        internal.boundsMin = glm::vec3(-1.0f);
        internal.boundsMax = glm::vec3(1.0f);
        internal.left = depth + 1 < internalCount ? depth + 1 : finalLeaf;
        internal.right = sideLeafStart + depth;
        internal.first = 0;
        internal.count = 0;

        checkpoint::BvhNodeRecord& sideLeaf =
            data.bvhNodes[static_cast<std::size_t>(sideLeafStart + depth)];
        sideLeaf.boundsMin = glm::vec3(-1.0f);
        sideLeaf.boundsMax = glm::vec3(1.0f);
        sideLeaf.first = depth;
        sideLeaf.count = 1;
    }
    checkpoint::BvhNodeRecord& last =
        data.bvhNodes[static_cast<std::size_t>(finalLeaf)];
    last.boundsMin = glm::vec3(-1.0f);
    last.boundsMax = glm::vec3(1.0f);
    last.first = internalCount;
    last.count = 1;
    expectMalformedBvh(data, "excessive traversal depth");
}

} // namespace

int main() {
    try {
        const std::uint64_t unique = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const std::filesystem::path directory =
            std::filesystem::temp_directory_path() /
            ("cis565-checkpoint-tests-" + std::to_string(unique));
        std::filesystem::create_directories(directory);
        DirectoryGuard cleanup(directory);

        std::size_t groups = 0;
        testRoundTrip(directory);
        ++groups;
        testFormatLayoutAndSamplerIds(directory);
        ++groups;
        testSamplerIdValidation(directory);
        ++groups;
        testCorruption(directory);
        ++groups;
        testContinuationMismatch(directory);
        ++groups;
        testSizeValidation(directory);
        ++groups;
        testNonFiniteValidation(directory);
        ++groups;
        testProjectionValidation(directory);
        ++groups;
        testShutterValidation(directory);
        ++groups;
        testRussianRouletteBounds();
        ++groups;
        testConfigEncodingRejection(directory);
        ++groups;
        testSchedulingConflictRejection(directory);
        ++groups;
        testOldVersionRejection(directory);
        ++groups;
        testFingerprinting(directory);
        ++groups;
        testBufferedBoundaries(directory);
        ++groups;
        testBvhValidation();
        ++groups;
        std::cout << "checkpoint_tests: PASS (" << groups << " groups, "
                  << gChecks << " checks)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "checkpoint_tests: FAIL after " << gChecks
                  << " checks: " << error.what() << '\n';
        return 1;
    }
}
