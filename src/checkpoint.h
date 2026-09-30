#pragma once

#include "glm/glm.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

// Lossless renderer checkpoints.  The on-disk representation is deliberately
// independent of C++ object layout, GLM alignment, pointer size, and host
// endianness: every integer and IEEE-754 float is written field-by-field in
// little-endian order.
namespace checkpoint {

inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::uint64_t kFnv1aOffsetBasis = 14695981039346656037ull;
inline constexpr std::uint64_t kFnv1aPrime = 1099511628211ull;

enum class Error {
    None = 0,
    InvalidArgument,
    Io,
    BadMagic,
    UnsupportedVersion,
    LimitExceeded,
    Overflow,
    Truncated,
    TrailingData,
    ChecksumMismatch,
    Malformed,
    NonFinite,
    ContinuationMismatch,
    AtomicReplaceFailed,
    OutOfMemory
};

struct Status {
    Error code = Error::None;
    std::string message;

    bool ok() const noexcept { return code == Error::None; }
    explicit operator bool() const noexcept { return ok(); }
};

// Target sample count, output path, checkpoint cadence, and benchmark settings
// intentionally do not appear here.  A resumed render may extend its target.
struct RenderConfig {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t maxDepth = 0;
    bool antialias = false;
    bool nextEventEstimation = false;
    bool useBvh = false;
    bool compact = false;
    bool materialSort = false;
};

struct CameraState {
    glm::vec3 position{0.0f};
    glm::vec3 lookAt{0.0f};
    glm::vec3 view{0.0f};
    glm::vec3 up{0.0f};
    glm::vec3 right{0.0f};
    glm::vec2 fov{0.0f};
    glm::vec2 pixelLength{0.0f};
    float aperture = 0.0f;
    float focusDistance = 1.0f;
};

// Exact continuation can depend on the generated machine code and GPU/runtime.
// Callers should populate all of these from their build and selected CUDA
// device.  They are checked byte-for-byte/numerically on resume when requested.
struct CompatibilityInfo {
    std::uint32_t cudaRuntimeVersion = 0;
    std::uint32_t cudaDriverVersion = 0;
    std::uint32_t computeCapabilityMajor = 0;
    std::uint32_t computeCapabilityMinor = 0;
    std::string deviceIdentifier;
    std::string sourceRevision;
    std::string buildIdentifier;
};

// These values are supplied by the caller after hashing canonical scene input.
// renderConfig must omit the extendable target/output/benchmark settings.
struct Fingerprints {
    std::uint64_t sceneFile = 0;
    std::uint64_t assets = 0;
    std::uint64_t materials = 0;
    std::uint64_t geometry = 0;
    std::uint64_t renderConfig = 0;
};

// DTO matching the renderer's indexed BVH without depending on CUDA or the
// scene/geometry headers.  Integration copies fields explicitly in both
// directions.
struct BvhNodeRecord {
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    std::int32_t left = -1;
    std::int32_t right = -1;
    std::int32_t first = 0;
    std::int32_t count = 0;
};

struct CheckpointData {
    // This is both the number of completed samples and the zero-based global
    // sample index to use next.
    std::uint64_t completedSamples = 0;
    std::uint64_t seed = 0;
    std::uint32_t samplerVersion = 0;
    RenderConfig config;
    CameraState camera;
    CompatibilityInfo compatibility;
    Fingerprints fingerprints;
    std::vector<glm::vec3> accumulation;
    std::vector<BvhNodeRecord> bvhNodes;
    std::vector<std::int32_t> bvhPrimitiveIndices;
};

// Bounded preflight result used to resolve options inherited from a resume
// file.  inspect() verifies the checksum, scalar metadata, counts, and encoded
// array lengths, but deliberately does not allocate or decode the arrays.
struct CheckpointMetadata {
    std::uint64_t completedSamples = 0;
    std::uint64_t seed = 0;
    std::uint32_t samplerVersion = 0;
    RenderConfig config;
    CameraState camera;
    CompatibilityInfo compatibility;
    Fingerprints fingerprints;
    std::uint64_t accumulationCount = 0;
    std::uint64_t bvhNodeCount = 0;
    std::uint64_t bvhPrimitiveIndexCount = 0;
};

struct ExpectedState {
    std::uint64_t seed = 0;
    std::uint32_t samplerVersion = 0;
    RenderConfig config;
    CameraState camera;
    CompatibilityInfo compatibility;
    Fingerprints fingerprints;
    bool requireSeedAndSampler = true;
    bool requireCompatibility = true;
};

// Defaults admit an 8K image while bounding every allocation derived from an
// untrusted file.  Applications may pass stricter or intentionally larger
// limits, subject to size_t and serialized-size overflow checks.
struct Limits {
    std::uint32_t maxWidth = 32768;
    std::uint32_t maxHeight = 32768;
    std::uint32_t maxTraceDepth = 4096;
    std::uint64_t maxPixels = 33554432ull;
    std::uint64_t maxBvhNodes = 33554432ull;
    std::uint64_t maxBvhIndices = 134217728ull;
    std::uint64_t maxCompletedSamples = 0x7fffffffffffffffull;
    std::uint64_t maxFileBytes = 4ull * 1024ull * 1024ull * 1024ull;
    std::uint32_t maxStringBytes = 1024u * 1024u;
};

Status validate(const CheckpointData& data, const Limits& limits = {});
Status validateForContinuation(const CheckpointData& data,
                               const ExpectedState& expected);

// Writes a sibling temporary file, flushes/closes it, then atomically renames
// it over path.  A failed write leaves an existing checkpoint untouched.
Status saveAtomic(const std::filesystem::path& path,
                  const CheckpointData& data,
                  const Limits& limits = {});

// Reads bounded metadata without allocating accumulation or BVH vectors.  A
// caller resuming a render should use this only to resolve inherited options,
// construct ExpectedState, and then call load(path, ..., &expected).
Status inspect(const std::filesystem::path& path,
               CheckpointMetadata& out,
               const Limits& limits = {});

// Integrity and scalar metadata are validated before image/BVH allocation.
// `out` is modified only after the complete file has passed validation.
Status load(const std::filesystem::path& path,
            CheckpointData& out,
            const ExpectedState* expected = nullptr,
            const Limits& limits = {});

// FNV-1a is used only for deterministic change detection and accidental
// corruption detection.  It is not a cryptographic hash and is not suitable
// for authenticating files supplied by an adversary.
std::uint64_t fnv1aUpdate(std::uint64_t state,
                          const void* bytes,
                          std::size_t byteCount) noexcept;

class FingerprintBuilder {
public:
    explicit FingerprintBuilder(std::uint64_t seed = kFnv1aOffsetBasis) noexcept;

    FingerprintBuilder& addBytes(const void* bytes, std::size_t byteCount) noexcept;
    FingerprintBuilder& addBool(bool value) noexcept;
    FingerprintBuilder& addU32(std::uint32_t value) noexcept;
    FingerprintBuilder& addI32(std::int32_t value) noexcept;
    FingerprintBuilder& addU64(std::uint64_t value) noexcept;
    FingerprintBuilder& addFloat(float value) noexcept;
    FingerprintBuilder& addString(std::string_view value) noexcept;
    FingerprintBuilder& addVec2(const glm::vec2& value) noexcept;
    FingerprintBuilder& addVec3(const glm::vec3& value) noexcept;

    std::uint64_t value() const noexcept { return state_; }

private:
    std::uint64_t state_;
};

// Hashes file contents without loading the complete file into memory.
Status fingerprintFile(const std::filesystem::path& path,
                       std::uint64_t& fingerprint,
                       std::uint64_t seed = kFnv1aOffsetBasis);

} // namespace checkpoint
