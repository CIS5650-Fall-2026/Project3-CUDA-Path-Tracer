#include "checkpoint.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace checkpoint {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'C', '5', '6', '5', 'C', 'K', 'P', 'T'};
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr std::uint64_t kHeaderBytes = 24;
constexpr std::uint64_t kChecksumBytes = 8;
constexpr std::size_t kIoBufferBytes = 64 * 1024;

static_assert(sizeof(float) == 4, "checkpoint format requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559,
              "checkpoint format requires IEEE-754 float");

Status failure(Error code, std::string message) {
    return Status{code, std::move(message)};
}

bool checkedAdd(std::uint64_t a, std::uint64_t b, std::uint64_t& result) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

bool checkedMultiply(std::uint64_t a,
                     std::uint64_t b,
                     std::uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

void encodeU32(std::uint32_t value, std::uint8_t* bytes) {
    for (unsigned i = 0; i < 4; ++i) {
        bytes[i] = static_cast<std::uint8_t>((value >> (8u * i)) & 0xffu);
    }
}

void encodeU64(std::uint64_t value, std::uint8_t* bytes) {
    for (unsigned i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::uint8_t>((value >> (8u * i)) & 0xffu);
    }
}

std::uint32_t decodeU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(bytes[i]) << (8u * i);
    }
    return value;
}

std::uint64_t decodeU64(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8u * i);
    }
    return value;
}

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float floatFromBits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool sameFloat(float a, float b) {
    return floatBits(a) == floatBits(b);
}

bool finite(float value) {
    return std::isfinite(static_cast<double>(value));
}

bool finite(const glm::vec2& value) {
    return finite(value.x) && finite(value.y);
}

bool finite(const glm::vec3& value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool sameVec2(const glm::vec2& a, const glm::vec2& b) {
    return sameFloat(a.x, b.x) && sameFloat(a.y, b.y);
}

bool sameVec3(const glm::vec3& a, const glm::vec3& b) {
    return sameFloat(a.x, b.x) && sameFloat(a.y, b.y) &&
           sameFloat(a.z, b.z);
}

Status validateLimits(const Limits& limits) {
    if (limits.maxWidth == 0 || limits.maxHeight == 0 ||
        limits.maxTraceDepth == 0 || limits.maxPixels == 0 ||
        limits.maxFileBytes < kHeaderBytes + kChecksumBytes ||
        limits.maxStringBytes == 0) {
        return failure(Error::InvalidArgument,
                       "checkpoint limits must be positive and internally valid");
    }
    return {};
}

Status validateMetadata(const CheckpointData& data, const Limits& limits) {
    Status status = validateLimits(limits);
    if (!status) {
        return status;
    }

    if (data.config.width == 0 || data.config.height == 0) {
        return failure(Error::Malformed,
                       "checkpoint resolution must be nonzero");
    }
    if (data.config.width > limits.maxWidth ||
        data.config.height > limits.maxHeight) {
        return failure(Error::LimitExceeded,
                       "checkpoint resolution exceeds configured limits");
    }
    if (data.config.maxDepth == 0) {
        return failure(Error::Malformed,
                       "checkpoint trace depth must be nonzero");
    }
    if (data.config.maxDepth > limits.maxTraceDepth) {
        return failure(Error::LimitExceeded,
                       "checkpoint trace depth exceeds configured limit");
    }
    if (data.completedSamples > limits.maxCompletedSamples) {
        return failure(Error::LimitExceeded,
                       "checkpoint completed-sample count exceeds configured limit");
    }

    if (!finite(data.camera.position) || !finite(data.camera.lookAt) ||
        !finite(data.camera.view) || !finite(data.camera.up) ||
        !finite(data.camera.right) || !finite(data.camera.fov) ||
        !finite(data.camera.pixelLength) || !finite(data.camera.aperture) ||
        !finite(data.camera.focusDistance)) {
        return failure(Error::NonFinite,
                       "checkpoint camera contains a non-finite value");
    }
    if (data.camera.aperture < 0.0f || data.camera.focusDistance <= 0.0f) {
        return failure(Error::Malformed,
                       "checkpoint camera has an invalid aperture or focus distance");
    }
    if (data.camera.fov.x <= 0.0f || data.camera.fov.y <= 0.0f ||
        data.camera.pixelLength.x <= 0.0f ||
        data.camera.pixelLength.y <= 0.0f) {
        return failure(Error::Malformed,
                       "checkpoint camera has invalid projection parameters");
    }

    const auto stringWithinLimit = [&limits](const std::string& value) {
        return value.size() <= limits.maxStringBytes &&
               value.size() <= std::numeric_limits<std::uint32_t>::max();
    };
    if (!stringWithinLimit(data.compatibility.deviceIdentifier) ||
        !stringWithinLimit(data.compatibility.sourceRevision) ||
        !stringWithinLimit(data.compatibility.buildIdentifier)) {
        return failure(Error::LimitExceeded,
                       "checkpoint compatibility string exceeds configured limit");
    }

    return {};
}

bool addPayloadBytes(std::uint64_t amount, std::uint64_t& total) {
    std::uint64_t updated = 0;
    if (!checkedAdd(total, amount, updated)) {
        return false;
    }
    total = updated;
    return true;
}

Status payloadSize(const CheckpointData& data,
                   const Limits& limits,
                   std::uint64_t& result) {
    std::uint64_t total = 0;
    // completedSamples, seed, samplerVersion
    if (!addPayloadBytes(8 + 8 + 4, total) ||
        // RenderConfig: three u32 values and five canonical booleans.
        !addPayloadBytes(3 * 4 + 5, total) ||
        // CameraState: twenty-one floats.
        !addPayloadBytes(21 * 4, total) ||
        // CompatibilityInfo numeric fields.
        !addPayloadBytes(4 * 4, total)) {
        return failure(Error::Overflow,
                       "checkpoint fixed metadata size overflow");
    }

    const std::array<const std::string*, 3> strings = {
        &data.compatibility.deviceIdentifier,
        &data.compatibility.sourceRevision,
        &data.compatibility.buildIdentifier};
    for (const std::string* value : strings) {
        if (!addPayloadBytes(4, total) ||
            !addPayloadBytes(static_cast<std::uint64_t>(value->size()), total)) {
            return failure(Error::Overflow,
                           "checkpoint compatibility string size overflow");
        }
    }

    // Five fingerprints followed by three vector counts.
    if (!addPayloadBytes(5 * 8, total) || !addPayloadBytes(3 * 8, total)) {
        return failure(Error::Overflow, "checkpoint metadata size overflow");
    }

    std::uint64_t bytes = 0;
    if (!checkedMultiply(static_cast<std::uint64_t>(data.accumulation.size()),
                         3 * 4, bytes) ||
        !addPayloadBytes(bytes, total) ||
        !checkedMultiply(static_cast<std::uint64_t>(data.bvhNodes.size()),
                         6 * 4 + 4 * 4, bytes) ||
        !addPayloadBytes(bytes, total) ||
        !checkedMultiply(
            static_cast<std::uint64_t>(data.bvhPrimitiveIndices.size()), 4,
            bytes) ||
        !addPayloadBytes(bytes, total)) {
        return failure(Error::Overflow, "checkpoint array size overflow");
    }

    std::uint64_t fileBytes = 0;
    if (!checkedAdd(kHeaderBytes, total, fileBytes) ||
        !checkedAdd(fileBytes, kChecksumBytes, fileBytes)) {
        return failure(Error::Overflow, "checkpoint file size overflow");
    }
    if (fileBytes > limits.maxFileBytes) {
        return failure(Error::LimitExceeded,
                       "checkpoint file exceeds configured size limit");
    }

    result = total;
    return {};
}

class BufferedWriter {
public:
    explicit BufferedWriter(const std::filesystem::path& path)
        : stream_(path, std::ios::binary | std::ios::trunc) {}

    bool isOpen() const { return stream_.is_open(); }
    bool good() const { return good_; }
    std::uint64_t byteCount() const { return byteCount_; }
    std::uint64_t hash() const { return hash_; }

    bool raw(const void* source, std::size_t size, bool includeInHash = true) {
        if (!good_ || (source == nullptr && size != 0)) {
            good_ = false;
            return false;
        }
        if (includeInHash) {
            hash_ = fnv1aUpdate(hash_, source, size);
        }
        if (size > std::numeric_limits<std::uint64_t>::max() - byteCount_) {
            good_ = false;
            return false;
        }
        byteCount_ += static_cast<std::uint64_t>(size);

        const auto* input = static_cast<const std::uint8_t*>(source);
        while (size != 0) {
            if (used_ == buffer_.size() && !flushBuffer()) {
                return false;
            }
            const std::size_t available = buffer_.size() - used_;
            const std::size_t amount = size < available ? size : available;
            std::memcpy(buffer_.data() + used_, input, amount);
            used_ += amount;
            input += amount;
            size -= amount;
        }
        return true;
    }

    bool u8(std::uint8_t value) { return raw(&value, sizeof(value)); }

    bool u32(std::uint32_t value) {
        std::uint8_t bytes[4];
        encodeU32(value, bytes);
        return raw(bytes, sizeof(bytes));
    }

    bool i32(std::int32_t value) {
        return u32(static_cast<std::uint32_t>(value));
    }

    bool u64(std::uint64_t value, bool includeInHash = true) {
        std::uint8_t bytes[8];
        encodeU64(value, bytes);
        return raw(bytes, sizeof(bytes), includeInHash);
    }

    bool f32(float value) { return u32(floatBits(value)); }

    bool vec2(const glm::vec2& value) {
        return f32(value.x) && f32(value.y);
    }

    bool vec3(const glm::vec3& value) {
        return f32(value.x) && f32(value.y) && f32(value.z);
    }

    bool string(const std::string& value) {
        return u32(static_cast<std::uint32_t>(value.size())) &&
               raw(value.data(), value.size());
    }

    bool close() {
        if (!flushBuffer()) {
            return false;
        }
        stream_.flush();
        if (!stream_) {
            good_ = false;
        }
        stream_.close();
        if (stream_.fail()) {
            good_ = false;
        }
        return good_;
    }

private:
    bool flushBuffer() {
        if (!good_) {
            return false;
        }
        if (used_ != 0) {
            stream_.write(reinterpret_cast<const char*>(buffer_.data()),
                          static_cast<std::streamsize>(used_));
            if (!stream_) {
                good_ = false;
                return false;
            }
            used_ = 0;
        }
        return true;
    }

    std::ofstream stream_;
    std::array<std::uint8_t, kIoBufferBytes> buffer_{};
    std::size_t used_ = 0;
    bool good_ = true;
    std::uint64_t byteCount_ = 0;
    std::uint64_t hash_ = kFnv1aOffsetBasis;
};

class BufferedReader {
public:
    BufferedReader(std::ifstream& stream, std::uint64_t remaining)
        : stream_(stream), remaining_(remaining) {}

    std::uint64_t remaining() const { return remaining_; }
    bool good() const { return good_; }

    bool raw(void* destination, std::size_t size) {
        if (!good_ || (destination == nullptr && size != 0) ||
            static_cast<std::uint64_t>(size) > remaining_) {
            good_ = false;
            return false;
        }
        auto* output = static_cast<std::uint8_t*>(destination);
        std::size_t needed = size;
        while (needed != 0) {
            if (position_ == available_ && !refill()) {
                return false;
            }
            const std::size_t buffered = available_ - position_;
            const std::size_t amount = needed < buffered ? needed : buffered;
            std::memcpy(output, buffer_.data() + position_, amount);
            output += amount;
            position_ += amount;
            needed -= amount;
            remaining_ -= static_cast<std::uint64_t>(amount);
        }
        return true;
    }

    bool u8(std::uint8_t& value) { return raw(&value, sizeof(value)); }

    bool u32(std::uint32_t& value) {
        std::uint8_t bytes[4];
        if (!raw(bytes, sizeof(bytes))) {
            return false;
        }
        value = decodeU32(bytes);
        return true;
    }

    bool i32(std::int32_t& value) {
        std::uint32_t encoded = 0;
        if (!u32(encoded)) {
            return false;
        }
        if (encoded <=
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            value = static_cast<std::int32_t>(encoded);
        } else {
            const std::uint32_t magnitude = (~encoded) + 1u;
            if (magnitude == 0x80000000u) {
                value = std::numeric_limits<std::int32_t>::min();
            } else {
                value = -static_cast<std::int32_t>(magnitude);
            }
        }
        return true;
    }

    bool u64(std::uint64_t& value) {
        std::uint8_t bytes[8];
        if (!raw(bytes, sizeof(bytes))) {
            return false;
        }
        value = decodeU64(bytes);
        return true;
    }

    bool f32(float& value) {
        std::uint32_t encoded = 0;
        if (!u32(encoded)) {
            return false;
        }
        value = floatFromBits(encoded);
        return true;
    }

    bool vec2(glm::vec2& value) {
        return f32(value.x) && f32(value.y);
    }

    bool vec3(glm::vec3& value) {
        return f32(value.x) && f32(value.y) && f32(value.z);
    }

    Status string(std::string& value, std::uint32_t maxBytes) {
        std::uint32_t size = 0;
        if (!u32(size)) {
            return failure(Error::Truncated,
                           "checkpoint ended in a string length");
        }
        if (size > maxBytes) {
            return failure(Error::LimitExceeded,
                           "checkpoint string exceeds configured limit");
        }
        if (static_cast<std::uint64_t>(size) > remaining_) {
            return failure(Error::Truncated,
                           "checkpoint ended in a string value");
        }
        value.resize(size);
        if (size != 0 && !raw(&value[0], size)) {
            return failure(Error::Truncated,
                           "checkpoint ended in a string value");
        }
        return {};
    }

private:
    bool refill() {
        if (remaining_ == 0) {
            good_ = false;
            return false;
        }
        const std::uint64_t requested64 =
            remaining_ < buffer_.size() ? remaining_ : buffer_.size();
        const std::streamsize requested =
            static_cast<std::streamsize>(requested64);
        stream_.read(reinterpret_cast<char*>(buffer_.data()), requested);
        if (stream_.gcount() != requested) {
            good_ = false;
            return false;
        }
        position_ = 0;
        available_ = static_cast<std::size_t>(requested);
        return true;
    }

    std::ifstream& stream_;
    std::array<std::uint8_t, kIoBufferBytes> buffer_{};
    std::size_t position_ = 0;
    std::size_t available_ = 0;
    std::uint64_t remaining_ = 0;
    bool good_ = true;
};

bool writeConfig(BufferedWriter& writer, const RenderConfig& config) {
    return writer.u32(config.width) && writer.u32(config.height) &&
           writer.u32(config.maxDepth) &&
           writer.u8(config.antialias ? 1u : 0u) &&
           writer.u8(config.nextEventEstimation ? 1u : 0u) &&
           writer.u8(config.useBvh ? 1u : 0u) &&
           writer.u8(config.compact ? 1u : 0u) &&
           writer.u8(config.materialSort ? 1u : 0u);
}

bool writeCamera(BufferedWriter& writer, const CameraState& camera) {
    return writer.vec3(camera.position) && writer.vec3(camera.lookAt) &&
           writer.vec3(camera.view) && writer.vec3(camera.up) &&
           writer.vec3(camera.right) && writer.vec2(camera.fov) &&
           writer.vec2(camera.pixelLength) && writer.f32(camera.aperture) &&
           writer.f32(camera.focusDistance);
}

bool writeCompatibility(BufferedWriter& writer,
                        const CompatibilityInfo& compatibility) {
    return writer.u32(compatibility.cudaRuntimeVersion) &&
           writer.u32(compatibility.cudaDriverVersion) &&
           writer.u32(compatibility.computeCapabilityMajor) &&
           writer.u32(compatibility.computeCapabilityMinor) &&
           writer.string(compatibility.deviceIdentifier) &&
           writer.string(compatibility.sourceRevision) &&
           writer.string(compatibility.buildIdentifier);
}

bool writeFingerprints(BufferedWriter& writer,
                       const Fingerprints& fingerprints) {
    return writer.u64(fingerprints.sceneFile) &&
           writer.u64(fingerprints.assets) &&
           writer.u64(fingerprints.materials) &&
           writer.u64(fingerprints.geometry) &&
           writer.u64(fingerprints.renderConfig);
}

bool readCanonicalBool(BufferedReader& reader, bool& output) {
    std::uint8_t value = 0;
    if (!reader.u8(value) || value > 1) {
        return false;
    }
    output = value != 0;
    return true;
}

bool readConfig(BufferedReader& reader, RenderConfig& config) {
    return reader.u32(config.width) && reader.u32(config.height) &&
           reader.u32(config.maxDepth) &&
           readCanonicalBool(reader, config.antialias) &&
           readCanonicalBool(reader, config.nextEventEstimation) &&
           readCanonicalBool(reader, config.useBvh) &&
           readCanonicalBool(reader, config.compact) &&
           readCanonicalBool(reader, config.materialSort);
}

bool readCamera(BufferedReader& reader, CameraState& camera) {
    return reader.vec3(camera.position) && reader.vec3(camera.lookAt) &&
           reader.vec3(camera.view) && reader.vec3(camera.up) &&
           reader.vec3(camera.right) && reader.vec2(camera.fov) &&
           reader.vec2(camera.pixelLength) && reader.f32(camera.aperture) &&
           reader.f32(camera.focusDistance);
}

Status readCompatibility(BufferedReader& reader,
                         CompatibilityInfo& compatibility,
                         const Limits& limits) {
    if (!reader.u32(compatibility.cudaRuntimeVersion) ||
        !reader.u32(compatibility.cudaDriverVersion) ||
        !reader.u32(compatibility.computeCapabilityMajor) ||
        !reader.u32(compatibility.computeCapabilityMinor)) {
        return failure(Error::Truncated,
                       "checkpoint ended in compatibility metadata");
    }
    Status status = reader.string(compatibility.deviceIdentifier,
                                  limits.maxStringBytes);
    if (!status) {
        return status;
    }
    status = reader.string(compatibility.sourceRevision,
                           limits.maxStringBytes);
    if (!status) {
        return status;
    }
    return reader.string(compatibility.buildIdentifier,
                         limits.maxStringBytes);
}

bool readFingerprints(BufferedReader& reader, Fingerprints& fingerprints) {
    return reader.u64(fingerprints.sceneFile) &&
           reader.u64(fingerprints.assets) &&
           reader.u64(fingerprints.materials) &&
           reader.u64(fingerprints.geometry) &&
           reader.u64(fingerprints.renderConfig);
}

std::filesystem::path makeTemporaryPath(const std::filesystem::path& target) {
    static std::atomic<std::uint64_t> sequence{0};
    const std::uint64_t timeValue = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::uint64_t sequenceValue =
        sequence.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream suffix;
    suffix << ".tmp-" << std::hex << timeValue << '-' << sequenceValue;
    std::filesystem::path name = target.filename();
    name += suffix.str();
    const std::filesystem::path parent =
        target.has_parent_path() ? target.parent_path() :
                                   std::filesystem::path(".");
    return parent / name;
}

Status atomicReplace(const std::filesystem::path& temporary,
                     const std::filesystem::path& target) {
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), target.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return failure(Error::AtomicReplaceFailed,
                       "could not atomically replace checkpoint (Windows error " +
                           std::to_string(GetLastError()) + ")");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        return failure(Error::AtomicReplaceFailed,
                       "could not atomically replace checkpoint: " +
                           error.message());
    }
#endif
    return {};
}

class TemporaryFileGuard {
public:
    explicit TemporaryFileGuard(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~TemporaryFileGuard() {
        if (!committed_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }
    void commit() { committed_ = true; }

private:
    std::filesystem::path path_;
    bool committed_ = false;
};

Status verifyChecksumAndHeader(std::ifstream& stream,
                               std::uint64_t fileBytes,
                               const Limits& limits,
                               std::uint64_t& payloadBytes) {
    if (fileBytes < kHeaderBytes + kChecksumBytes) {
        return failure(Error::Truncated,
                       "checkpoint is shorter than its fixed header");
    }
    if (fileBytes > limits.maxFileBytes) {
        return failure(Error::LimitExceeded,
                       "checkpoint exceeds configured file-size limit");
    }

    std::array<std::uint8_t, kHeaderBytes> header{};
    stream.read(reinterpret_cast<char*>(header.data()),
                static_cast<std::streamsize>(header.size()));
    if (stream.gcount() != static_cast<std::streamsize>(header.size())) {
        return failure(Error::Truncated,
                       "checkpoint ended while reading its header");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
        return failure(Error::BadMagic, "checkpoint magic does not match");
    }
    const std::uint32_t version = decodeU32(header.data() + 8);
    if (version != kFormatVersion) {
        return failure(Error::UnsupportedVersion,
                       "unsupported checkpoint format version " +
                           std::to_string(version));
    }
    if (decodeU32(header.data() + 12) != kEndianMarker) {
        return failure(Error::Malformed,
                       "checkpoint encoding marker is invalid");
    }
    payloadBytes = decodeU64(header.data() + 16);

    std::uint64_t expectedBytes = 0;
    if (!checkedAdd(kHeaderBytes, payloadBytes, expectedBytes) ||
        !checkedAdd(expectedBytes, kChecksumBytes, expectedBytes)) {
        return failure(Error::Overflow,
                       "checkpoint encoded file size overflows");
    }
    if (fileBytes < expectedBytes) {
        return failure(Error::Truncated,
                       "checkpoint is shorter than its encoded payload size");
    }
    if (fileBytes > expectedBytes) {
        return failure(Error::TrailingData,
                       "checkpoint contains bytes after its checksum");
    }

    stream.clear();
    stream.seekg(0, std::ios::beg);
    if (!stream) {
        return failure(Error::Io, "could not seek in checkpoint");
    }
    std::array<std::uint8_t, kIoBufferBytes> buffer{};
    std::uint64_t remaining = kHeaderBytes + payloadBytes;
    std::uint64_t hash = kFnv1aOffsetBasis;
    while (remaining != 0) {
        const std::size_t amount = static_cast<std::size_t>(
            remaining < buffer.size() ? remaining : buffer.size());
        stream.read(reinterpret_cast<char*>(buffer.data()),
                    static_cast<std::streamsize>(amount));
        if (stream.gcount() != static_cast<std::streamsize>(amount)) {
            return failure(Error::Truncated,
                           "checkpoint ended during checksum validation");
        }
        hash = fnv1aUpdate(hash, buffer.data(), amount);
        remaining -= amount;
    }
    std::uint8_t checksumBytes[8];
    stream.read(reinterpret_cast<char*>(checksumBytes), sizeof(checksumBytes));
    if (stream.gcount() != static_cast<std::streamsize>(sizeof(checksumBytes))) {
        return failure(Error::Truncated,
                       "checkpoint ended while reading its checksum");
    }
    if (hash != decodeU64(checksumBytes)) {
        return failure(Error::ChecksumMismatch,
                       "checkpoint checksum does not match its contents");
    }
    return {};
}

Status prepareInput(const std::filesystem::path& path,
                    const Limits& limits,
                    std::ifstream& stream,
                    std::uint64_t& payloadBytes) {
    Status status = validateLimits(limits);
    if (!status) {
        return status;
    }
    if (path.empty()) {
        return failure(Error::InvalidArgument,
                       "checkpoint input path is empty");
    }
    std::error_code sizeError;
    const std::uintmax_t diskSize = std::filesystem::file_size(path, sizeError);
    if (sizeError) {
        return failure(Error::Io,
                       "could not determine checkpoint size: " +
                           sizeError.message());
    }
    if (diskSize > std::numeric_limits<std::uint64_t>::max()) {
        return failure(Error::LimitExceeded,
                       "checkpoint size is not representable");
    }
    const std::uint64_t fileBytes = static_cast<std::uint64_t>(diskSize);
    stream.open(path, std::ios::binary);
    if (!stream.is_open()) {
        return failure(Error::Io, "could not open checkpoint for reading");
    }
    status = verifyChecksumAndHeader(stream, fileBytes, limits, payloadBytes);
    if (!status) {
        return status;
    }
    stream.clear();
    stream.seekg(static_cast<std::streamoff>(kHeaderBytes), std::ios::beg);
    if (!stream) {
        return failure(Error::Io, "could not seek to checkpoint payload");
    }
    return {};
}

Status readCheckpointPrefix(BufferedReader& reader,
                            CheckpointData& loaded,
                            const ExpectedState* expected,
                            const Limits& limits,
                            std::uint64_t& pixelCount,
                            std::uint64_t& nodeCount,
                            std::uint64_t& indexCount) {
    if (!reader.u64(loaded.completedSamples) || !reader.u64(loaded.seed) ||
        !reader.u32(loaded.samplerVersion)) {
        return failure(Error::Truncated,
                       "checkpoint ended in sample metadata");
    }
    if (!readConfig(reader, loaded.config)) {
        return failure(Error::Malformed,
                       "checkpoint render configuration is truncated or non-canonical");
    }
    if (!readCamera(reader, loaded.camera)) {
        return failure(Error::Truncated,
                       "checkpoint ended in camera metadata");
    }
    Status status = readCompatibility(reader, loaded.compatibility, limits);
    if (!status) {
        return status;
    }
    if (!readFingerprints(reader, loaded.fingerprints)) {
        return failure(Error::Truncated,
                       "checkpoint ended in fingerprint metadata");
    }

    // This includes all finite/range checks on scalar metadata and occurs
    // before any image or BVH allocation.
    status = validateMetadata(loaded, limits);
    if (!status) {
        return status;
    }
    if (expected != nullptr) {
        status = validateForContinuation(loaded, *expected);
        if (!status) {
            return status;
        }
    }

    if (!reader.u64(pixelCount) || !reader.u64(nodeCount) ||
        !reader.u64(indexCount)) {
        return failure(Error::Truncated,
                       "checkpoint ended in array counts");
    }
    std::uint64_t expectedPixels = 0;
    if (!checkedMultiply(loaded.config.width, loaded.config.height,
                         expectedPixels)) {
        return failure(Error::Overflow, "checkpoint pixel count overflows");
    }
    if (pixelCount != expectedPixels) {
        return failure(Error::Malformed,
                       "checkpoint pixel count does not match resolution");
    }
    if (pixelCount > limits.maxPixels || nodeCount > limits.maxBvhNodes ||
        indexCount > limits.maxBvhIndices) {
        return failure(Error::LimitExceeded,
                       "checkpoint array count exceeds configured limit");
    }
    if (pixelCount > std::vector<glm::vec3>().max_size() ||
        nodeCount > std::vector<BvhNodeRecord>().max_size() ||
        indexCount > std::vector<std::int32_t>().max_size()) {
        return failure(Error::LimitExceeded,
                       "checkpoint array count exceeds host container capacity");
    }

    std::uint64_t expectedArrayBytes = 0;
    std::uint64_t bytes = 0;
    if (!checkedMultiply(pixelCount, 3 * 4, bytes) ||
        !addPayloadBytes(bytes, expectedArrayBytes) ||
        !checkedMultiply(nodeCount, 6 * 4 + 4 * 4, bytes) ||
        !addPayloadBytes(bytes, expectedArrayBytes) ||
        !checkedMultiply(indexCount, 4, bytes) ||
        !addPayloadBytes(bytes, expectedArrayBytes)) {
        return failure(Error::Overflow,
                       "checkpoint encoded array byte count overflows");
    }
    if (reader.remaining() < expectedArrayBytes) {
        return failure(Error::Truncated,
                       "checkpoint payload is shorter than its array counts");
    }
    if (reader.remaining() > expectedArrayBytes) {
        return failure(Error::TrailingData,
                       "checkpoint payload has unrecognized fields or trailing bytes");
    }
    return {};
}

std::string configMismatch(const RenderConfig& actual,
                           const RenderConfig& expected) {
    if (actual.width != expected.width || actual.height != expected.height) {
        return "resolution";
    }
    if (actual.maxDepth != expected.maxDepth) return "trace depth";
    if (actual.antialias != expected.antialias) return "AA setting";
    if (actual.nextEventEstimation != expected.nextEventEstimation) {
        return "NEE setting";
    }
    if (actual.useBvh != expected.useBvh) return "BVH setting";
    if (actual.compact != expected.compact) return "compaction setting";
    if (actual.materialSort != expected.materialSort) {
        return "material-sort setting";
    }
    return {};
}

bool sameCamera(const CameraState& a, const CameraState& b) {
    return sameVec3(a.position, b.position) && sameVec3(a.lookAt, b.lookAt) &&
           sameVec3(a.view, b.view) && sameVec3(a.up, b.up) &&
           sameVec3(a.right, b.right) && sameVec2(a.fov, b.fov) &&
           sameVec2(a.pixelLength, b.pixelLength) &&
           sameFloat(a.aperture, b.aperture) &&
           sameFloat(a.focusDistance, b.focusDistance);
}

bool sameCompatibility(const CompatibilityInfo& a,
                       const CompatibilityInfo& b) {
    return a.cudaRuntimeVersion == b.cudaRuntimeVersion &&
           a.cudaDriverVersion == b.cudaDriverVersion &&
           a.computeCapabilityMajor == b.computeCapabilityMajor &&
           a.computeCapabilityMinor == b.computeCapabilityMinor &&
           a.deviceIdentifier == b.deviceIdentifier &&
           a.sourceRevision == b.sourceRevision &&
           a.buildIdentifier == b.buildIdentifier;
}

} // namespace

std::uint64_t fnv1aUpdate(std::uint64_t state,
                          const void* bytes,
                          std::size_t byteCount) noexcept {
    if (bytes == nullptr) {
        return byteCount == 0 ? state : 0;
    }
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t i = 0; i < byteCount; ++i) {
        state ^= values[i];
        state *= kFnv1aPrime;
    }
    return state;
}

FingerprintBuilder::FingerprintBuilder(std::uint64_t seed) noexcept
    : state_(seed) {}

FingerprintBuilder& FingerprintBuilder::addBytes(const void* bytes,
                                                  std::size_t byteCount) noexcept {
    state_ = fnv1aUpdate(state_, bytes, byteCount);
    return *this;
}

FingerprintBuilder& FingerprintBuilder::addBool(bool value) noexcept {
    const std::uint8_t encoded = value ? 1u : 0u;
    return addBytes(&encoded, sizeof(encoded));
}

FingerprintBuilder& FingerprintBuilder::addU32(std::uint32_t value) noexcept {
    std::uint8_t encoded[4];
    encodeU32(value, encoded);
    return addBytes(encoded, sizeof(encoded));
}

FingerprintBuilder& FingerprintBuilder::addI32(std::int32_t value) noexcept {
    return addU32(static_cast<std::uint32_t>(value));
}

FingerprintBuilder& FingerprintBuilder::addU64(std::uint64_t value) noexcept {
    std::uint8_t encoded[8];
    encodeU64(value, encoded);
    return addBytes(encoded, sizeof(encoded));
}

FingerprintBuilder& FingerprintBuilder::addFloat(float value) noexcept {
    return addU32(floatBits(value));
}

FingerprintBuilder& FingerprintBuilder::addString(std::string_view value) noexcept {
    addU64(static_cast<std::uint64_t>(value.size()));
    return addBytes(value.data(), value.size());
}

FingerprintBuilder& FingerprintBuilder::addVec2(const glm::vec2& value) noexcept {
    return addFloat(value.x).addFloat(value.y);
}

FingerprintBuilder& FingerprintBuilder::addVec3(const glm::vec3& value) noexcept {
    return addFloat(value.x).addFloat(value.y).addFloat(value.z);
}

Status validate(const CheckpointData& data, const Limits& limits) {
    try {
        Status status = validateMetadata(data, limits);
        if (!status) {
            return status;
        }

        std::uint64_t pixelCount = 0;
        if (!checkedMultiply(data.config.width, data.config.height, pixelCount)) {
            return failure(Error::Overflow,
                           "checkpoint pixel count overflows");
        }
        if (pixelCount > limits.maxPixels) {
            return failure(Error::LimitExceeded,
                           "checkpoint pixel count exceeds configured limit");
        }
        if (pixelCount != static_cast<std::uint64_t>(data.accumulation.size())) {
            return failure(Error::Malformed,
                           "checkpoint accumulation size does not match resolution");
        }
        if (static_cast<std::uint64_t>(data.bvhNodes.size()) >
            limits.maxBvhNodes) {
            return failure(Error::LimitExceeded,
                           "checkpoint BVH-node count exceeds configured limit");
        }
        if (static_cast<std::uint64_t>(data.bvhPrimitiveIndices.size()) >
            limits.maxBvhIndices) {
            return failure(Error::LimitExceeded,
                           "checkpoint BVH-index count exceeds configured limit");
        }

        for (std::size_t i = 0; i < data.accumulation.size(); ++i) {
            if (!finite(data.accumulation[i])) {
                return failure(Error::NonFinite,
                               "checkpoint accumulation contains a non-finite value at pixel " +
                                   std::to_string(i));
            }
        }

        const std::uint64_t nodeCount = data.bvhNodes.size();
        const std::uint64_t indexCount = data.bvhPrimitiveIndices.size();
        if ((nodeCount == 0) != (indexCount == 0)) {
            return failure(Error::Malformed,
                           "checkpoint BVH nodes and primitive indices must both be empty or nonempty");
        }
        for (std::size_t i = 0; i < data.bvhNodes.size(); ++i) {
            const BvhNodeRecord& node = data.bvhNodes[i];
            if (!finite(node.boundsMin) || !finite(node.boundsMax)) {
                return failure(Error::NonFinite,
                               "checkpoint BVH node contains a non-finite bound at node " +
                                   std::to_string(i));
            }
            if (node.boundsMin.x > node.boundsMax.x ||
                node.boundsMin.y > node.boundsMax.y ||
                node.boundsMin.z > node.boundsMax.z) {
                return failure(Error::Malformed,
                               "checkpoint BVH node has inverted bounds at node " +
                                   std::to_string(i));
            }
            const auto childValid = [nodeCount](std::int32_t child) {
                return child == -1 ||
                       (child >= 0 && static_cast<std::uint64_t>(child) < nodeCount);
            };
            if (!childValid(node.left) || !childValid(node.right) ||
                node.left == static_cast<std::int32_t>(i) ||
                node.right == static_cast<std::int32_t>(i) ||
                (node.left >= 0 && node.left == node.right)) {
                return failure(Error::Malformed,
                               "checkpoint BVH node has an invalid child index at node " +
                                   std::to_string(i));
            }
            if (node.count < 0 || node.first < 0) {
                return failure(Error::Malformed,
                               "checkpoint BVH node has an invalid primitive range at node " +
                                   std::to_string(i));
            }
            if (node.count > 0) {
                if (node.left != -1 || node.right != -1) {
                    return failure(Error::Malformed,
                                   "checkpoint BVH leaf has child nodes at node " +
                                       std::to_string(i));
                }
                const std::uint64_t first =
                    static_cast<std::uint64_t>(node.first);
                const std::uint64_t count =
                    static_cast<std::uint64_t>(node.count);
                if (first > indexCount || count > indexCount - first) {
                    return failure(Error::Malformed,
                                   "checkpoint BVH node primitive range is out of bounds at node " +
                                       std::to_string(i));
                }
            } else if (node.left < 0 || node.right < 0) {
                return failure(Error::Malformed,
                               "checkpoint BVH internal node is missing a child at node " +
                                   std::to_string(i));
            }
        }

        // Validate the exact tree/range shape consumed by geometry.cuh.  This
        // prevents a checksummed but malformed checkpoint from introducing a
        // cycle, shared child, excessive traversal depth, or overlapping leaf
        // ranges on the GPU.
        if (nodeCount != 0) {
            std::vector<std::uint8_t> visited(
                static_cast<std::size_t>(nodeCount), 0u);
            std::vector<std::uint8_t> covered(
                static_cast<std::size_t>(indexCount), 0u);
            std::vector<std::pair<std::int32_t, std::uint32_t>> stack;
            stack.reserve(static_cast<std::size_t>(nodeCount));
            stack.emplace_back(0, 0);
            while (!stack.empty()) {
                const auto entry = stack.back();
                stack.pop_back();
                const std::size_t nodeIndex =
                    static_cast<std::size_t>(entry.first);
                if (visited[nodeIndex] != 0u) {
                    return failure(Error::Malformed,
                                   "checkpoint BVH contains a cycle or shared child");
                }
                if (entry.second > 60u) {
                    return failure(Error::Malformed,
                                   "checkpoint BVH exceeds renderer traversal depth");
                }
                visited[nodeIndex] = 1u;
                const BvhNodeRecord& node = data.bvhNodes[nodeIndex];
                if (node.count > 0) {
                    const std::size_t first =
                        static_cast<std::size_t>(node.first);
                    const std::size_t count =
                        static_cast<std::size_t>(node.count);
                    for (std::size_t offset = 0; offset < count; ++offset) {
                        if (covered[first + offset] != 0u) {
                            return failure(
                                Error::Malformed,
                                "checkpoint BVH leaf primitive ranges overlap");
                        }
                        covered[first + offset] = 1u;
                    }
                } else {
                    stack.emplace_back(node.right, entry.second + 1u);
                    stack.emplace_back(node.left, entry.second + 1u);
                }
            }
            for (std::uint8_t value : visited) {
                if (value == 0u) {
                    return failure(Error::Malformed,
                                   "checkpoint BVH contains an unreachable node");
                }
            }
            for (std::uint8_t value : covered) {
                if (value == 0u) {
                    return failure(Error::Malformed,
                                   "checkpoint BVH leaves do not cover every primitive index");
                }
            }
        }

        std::vector<std::uint8_t> primitiveSeen(
            static_cast<std::size_t>(indexCount), 0u);
        for (std::size_t i = 0; i < data.bvhPrimitiveIndices.size(); ++i) {
            const std::int32_t primitive = data.bvhPrimitiveIndices[i];
            if (primitive < 0 ||
                static_cast<std::uint64_t>(primitive) >= indexCount) {
                return failure(Error::Malformed,
                               "checkpoint contains an out-of-range primitive index at entry " +
                                   std::to_string(i));
            }
            const std::size_t primitiveIndex =
                static_cast<std::size_t>(primitive);
            if (primitiveSeen[primitiveIndex] != 0u) {
                return failure(Error::Malformed,
                               "checkpoint contains a duplicate primitive index");
            }
            primitiveSeen[primitiveIndex] = 1u;
        }

        std::uint64_t ignored = 0;
        return payloadSize(data, limits, ignored);
    } catch (const std::bad_alloc&) {
        return failure(Error::OutOfMemory,
                       "out of memory while validating checkpoint");
    } catch (const std::exception& error) {
        return failure(Error::InvalidArgument,
                       std::string("checkpoint validation failed: ") + error.what());
    } catch (...) {
        return failure(Error::InvalidArgument,
                       "checkpoint validation failed with an unknown error");
    }
}

Status validateForContinuation(const CheckpointData& data,
                               const ExpectedState& expected) {
    try {
        if (expected.requireSeedAndSampler) {
            if (data.seed != expected.seed) {
                return failure(Error::ContinuationMismatch,
                               "checkpoint seed does not match requested continuation");
            }
            if (data.samplerVersion != expected.samplerVersion) {
                return failure(
                    Error::ContinuationMismatch,
                    "checkpoint sampler version does not match this renderer");
            }
        }
        const std::string mismatch = configMismatch(data.config, expected.config);
        if (!mismatch.empty()) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint " + mismatch +
                               " does not match requested continuation");
        }
        if (!sameCamera(data.camera, expected.camera)) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint camera does not exactly match requested continuation");
        }
        if (data.fingerprints.sceneFile != expected.fingerprints.sceneFile) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint scene-file fingerprint mismatch");
        }
        if (data.fingerprints.assets != expected.fingerprints.assets) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint asset fingerprint mismatch");
        }
        if (data.fingerprints.materials != expected.fingerprints.materials) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint material fingerprint mismatch");
        }
        if (data.fingerprints.geometry != expected.fingerprints.geometry) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint geometry fingerprint mismatch");
        }
        if (data.fingerprints.renderConfig !=
            expected.fingerprints.renderConfig) {
            return failure(Error::ContinuationMismatch,
                           "checkpoint renderer-config fingerprint mismatch");
        }
        if (expected.requireCompatibility &&
            !sameCompatibility(data.compatibility, expected.compatibility)) {
            return failure(
                Error::ContinuationMismatch,
                "checkpoint GPU/runtime/build compatibility metadata mismatch");
        }
        return {};
    } catch (const std::bad_alloc&) {
        return failure(Error::OutOfMemory,
                       "out of memory while comparing checkpoint metadata");
    } catch (const std::exception& error) {
        return failure(Error::InvalidArgument,
                       std::string("checkpoint comparison failed: ") + error.what());
    } catch (...) {
        return failure(Error::InvalidArgument,
                       "checkpoint comparison failed with an unknown error");
    }
}

Status saveAtomic(const std::filesystem::path& path,
                  const CheckpointData& data,
                  const Limits& limits) {
    try {
        if (path.empty() || path.filename().empty()) {
            return failure(Error::InvalidArgument,
                           "checkpoint output path is empty or has no filename");
        }
        Status status = validate(data, limits);
        if (!status) {
            return status;
        }
        std::uint64_t payloadBytes = 0;
        status = payloadSize(data, limits, payloadBytes);
        if (!status) {
            return status;
        }

        std::filesystem::path temporary;
        bool foundName = false;
        for (unsigned attempt = 0; attempt < 64; ++attempt) {
            temporary = makeTemporaryPath(path);
            std::error_code error;
            const bool exists = std::filesystem::exists(temporary, error);
            if (error) {
                return failure(Error::Io,
                               "could not inspect checkpoint temporary path: " +
                                   error.message());
            }
            if (!exists) {
                foundName = true;
                break;
            }
        }
        if (!foundName) {
            return failure(Error::Io,
                           "could not choose a unique checkpoint temporary path");
        }
        TemporaryFileGuard temporaryGuard(temporary);
        BufferedWriter writer(temporary);
        if (!writer.isOpen()) {
            return failure(Error::Io,
                           "could not create checkpoint temporary file");
        }

        bool wrote = writer.raw(kMagic.data(), kMagic.size()) &&
                     writer.u32(kFormatVersion) && writer.u32(kEndianMarker) &&
                     writer.u64(payloadBytes) &&
                     writer.u64(data.completedSamples) && writer.u64(data.seed) &&
                     writer.u32(data.samplerVersion) &&
                     writeConfig(writer, data.config) &&
                     writeCamera(writer, data.camera) &&
                     writeCompatibility(writer, data.compatibility) &&
                     writeFingerprints(writer, data.fingerprints) &&
                     writer.u64(data.accumulation.size()) &&
                     writer.u64(data.bvhNodes.size()) &&
                     writer.u64(data.bvhPrimitiveIndices.size());

        for (const glm::vec3& pixel : data.accumulation) {
            wrote = wrote && writer.vec3(pixel);
        }
        for (const BvhNodeRecord& node : data.bvhNodes) {
            wrote = wrote && writer.vec3(node.boundsMin) &&
                    writer.vec3(node.boundsMax) && writer.i32(node.left) &&
                    writer.i32(node.right) && writer.i32(node.first) &&
                    writer.i32(node.count);
        }
        for (std::int32_t primitive : data.bvhPrimitiveIndices) {
            wrote = wrote && writer.i32(primitive);
        }

        if (!wrote || !writer.good()) {
            return failure(Error::Io,
                           "failed while writing checkpoint temporary file");
        }
        if (writer.byteCount() != kHeaderBytes + payloadBytes) {
            return failure(Error::Malformed,
                           "internal checkpoint serialized-size disagreement");
        }
        const std::uint64_t checksum = writer.hash();
        if (!writer.u64(checksum, false) || !writer.close()) {
            return failure(Error::Io,
                           "failed to flush checkpoint temporary file");
        }

        status = atomicReplace(temporary, path);
        if (!status) {
            return status;
        }
        temporaryGuard.commit();
        return {};
    } catch (const std::bad_alloc&) {
        return failure(Error::OutOfMemory,
                       "out of memory while saving checkpoint");
    } catch (const std::exception& error) {
        return failure(Error::Io,
                       std::string("could not save checkpoint: ") + error.what());
    } catch (...) {
        return failure(Error::Io,
                       "could not save checkpoint due to an unknown error");
    }
}

Status inspect(const std::filesystem::path& path,
               CheckpointMetadata& out,
               const Limits& limits) {
    try {
        std::ifstream stream;
        std::uint64_t payloadBytes = 0;
        Status status = prepareInput(path, limits, stream, payloadBytes);
        if (!status) return status;
        BufferedReader reader(stream, payloadBytes);
        CheckpointData loaded;
        std::uint64_t pixelCount = 0;
        std::uint64_t nodeCount = 0;
        std::uint64_t indexCount = 0;
        status = readCheckpointPrefix(reader, loaded, nullptr, limits,
                                      pixelCount, nodeCount, indexCount);
        if (!status) return status;

        CheckpointMetadata metadata;
        metadata.completedSamples = loaded.completedSamples;
        metadata.seed = loaded.seed;
        metadata.samplerVersion = loaded.samplerVersion;
        metadata.config = loaded.config;
        metadata.camera = loaded.camera;
        metadata.compatibility = std::move(loaded.compatibility);
        metadata.fingerprints = loaded.fingerprints;
        metadata.accumulationCount = pixelCount;
        metadata.bvhNodeCount = nodeCount;
        metadata.bvhPrimitiveIndexCount = indexCount;
        out = std::move(metadata);
        return {};
    } catch (const std::bad_alloc&) {
        return failure(Error::OutOfMemory,
                       "out of memory while inspecting checkpoint metadata");
    } catch (const std::length_error&) {
        return failure(Error::LimitExceeded,
                       "checkpoint metadata has an unrepresentable size");
    } catch (const std::exception& error) {
        return failure(Error::Io,
                       std::string("could not inspect checkpoint: ") + error.what());
    } catch (...) {
        return failure(Error::Io,
                       "could not inspect checkpoint due to an unknown error");
    }
}

Status load(const std::filesystem::path& path,
            CheckpointData& out,
            const ExpectedState* expected,
            const Limits& limits) {
    try {
        std::ifstream stream;
        std::uint64_t payloadBytes = 0;
        Status status = prepareInput(path, limits, stream, payloadBytes);
        if (!status) return status;
        BufferedReader reader(stream, payloadBytes);
        CheckpointData loaded;
        std::uint64_t pixelCount = 0;
        std::uint64_t nodeCount = 0;
        std::uint64_t indexCount = 0;
        status = readCheckpointPrefix(reader, loaded, expected, limits,
                                      pixelCount, nodeCount, indexCount);
        if (!status) return status;

        loaded.accumulation.resize(static_cast<std::size_t>(pixelCount));
        loaded.bvhNodes.resize(static_cast<std::size_t>(nodeCount));
        loaded.bvhPrimitiveIndices.resize(static_cast<std::size_t>(indexCount));
        for (glm::vec3& pixel : loaded.accumulation) {
            if (!reader.vec3(pixel)) {
                return failure(Error::Truncated,
                               "checkpoint ended in accumulation data");
            }
        }
        for (BvhNodeRecord& node : loaded.bvhNodes) {
            if (!reader.vec3(node.boundsMin) || !reader.vec3(node.boundsMax) ||
                !reader.i32(node.left) || !reader.i32(node.right) ||
                !reader.i32(node.first) || !reader.i32(node.count)) {
                return failure(Error::Truncated,
                               "checkpoint ended in BVH-node data");
            }
        }
        for (std::int32_t& primitive : loaded.bvhPrimitiveIndices) {
            if (!reader.i32(primitive)) {
                return failure(Error::Truncated,
                               "checkpoint ended in BVH-index data");
            }
        }
        if (reader.remaining() != 0 || !reader.good()) {
            return failure(Error::TrailingData,
                           "checkpoint payload was not consumed exactly");
        }

        status = validate(loaded, limits);
        if (!status) {
            return status;
        }
        out = std::move(loaded);
        return {};
    } catch (const std::bad_alloc&) {
        return failure(Error::OutOfMemory,
                       "out of memory while loading checkpoint");
    } catch (const std::length_error&) {
        return failure(Error::LimitExceeded,
                       "checkpoint requested an unrepresentable allocation");
    } catch (const std::exception& error) {
        return failure(Error::Io,
                       std::string("could not load checkpoint: ") + error.what());
    } catch (...) {
        return failure(Error::Io,
                       "could not load checkpoint due to an unknown error");
    }
}

Status fingerprintFile(const std::filesystem::path& path,
                       std::uint64_t& fingerprint,
                       std::uint64_t seed) {
    try {
        std::ifstream stream(path, std::ios::binary);
        if (!stream.is_open()) {
            return failure(Error::Io,
                           "could not open file for fingerprinting");
        }
        std::array<char, kIoBufferBytes> buffer{};
        std::uint64_t hash = seed;
        while (stream) {
            stream.read(buffer.data(),
                        static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = stream.gcount();
            if (count > 0) {
                hash = fnv1aUpdate(hash, buffer.data(),
                                   static_cast<std::size_t>(count));
            }
        }
        if (!stream.eof()) {
            return failure(Error::Io,
                           "I/O error while fingerprinting file");
        }
        fingerprint = hash;
        return {};
    } catch (const std::bad_alloc&) {
        return failure(Error::OutOfMemory,
                       "out of memory while fingerprinting file");
    } catch (const std::exception& error) {
        return failure(Error::Io,
                       std::string("could not fingerprint file: ") + error.what());
    } catch (...) {
        return failure(Error::Io,
                       "could not fingerprint file due to an unknown error");
    }
}

} // namespace checkpoint
