#include "render_checkpoint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace checkpoint {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Checkpoints require IEEE-754 binary32 float");
constexpr unsigned char magic[8] = {'P', 'T', 'C', 'H', 'K', 'P', 'T', 0};
constexpr std::uint32_t maxStringBytes = 16u * 1024u * 1024u;

[[noreturn]] void fail(const char* reason) { throw std::runtime_error(reason); }

std::uint64_t checksum(const unsigned char* bytes, std::size_t n)
{
    // FNV-1a detects accidental corruption; identity checks compare full content.
    std::uint64_t h = 14695981039346656037ull;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ bytes[i]) * 1099511628211ull;
    return h;
}

struct Writer {
    std::vector<unsigned char> bytes;
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<unsigned char>(v >> (8 * i)));
    }
    void u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<unsigned char>(v >> (8 * i)));
    }
    void f32(float v) {
        std::uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        u32(bits);
    }
    template<std::size_t N> void floats(const std::array<float, N>& v) {
        for (float x : v) f32(x);
    }
    void string(const std::string& v) {
        if (v.size() > maxStringBytes) fail("Checkpoint identity is too large");
        u32(static_cast<std::uint32_t>(v.size()));
        bytes.insert(bytes.end(), v.begin(), v.end());
    }
};

struct Reader {
    const std::vector<unsigned char>& bytes;
    std::size_t pos = 0;
    void need(std::size_t n) const {
        if (n > bytes.size() - pos) fail("Truncated checkpoint");
    }
    std::uint32_t u32() {
        need(4);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= std::uint32_t(bytes[pos++]) << (8 * i);
        return v;
    }
    std::uint64_t u64() {
        need(8);
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= std::uint64_t(bytes[pos++]) << (8 * i);
        return v;
    }
    float f32() {
        const auto bits = u32();
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    template<std::size_t N> void floats(std::array<float, N>& v) {
        for (float& x : v) x = f32();
    }
    std::string string() {
        const auto n = u32();
        if (n > maxStringBytes) fail("Checkpoint identity is too large");
        need(n);
        std::string result(bytes.begin() + pos, bytes.begin() + pos + n);
        pos += n;
        return result;
    }
};

template<std::size_t N> bool finite(const std::array<float, N>& v)
{
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

void validate(const Data& d)
{
    const auto pixels = std::uint64_t(d.width) * d.height;
    if (!d.width || !d.height || pixels > std::numeric_limits<int>::max() / 4 ||
        !d.completedSamples || d.completedSamples > std::numeric_limits<int>::max() ||
        d.traceDepth > std::numeric_limits<int>::max())
        fail("Invalid checkpoint dimensions, depth, or completed count");
    if (d.accumulation.size() != pixels * 3) fail("Checkpoint RGB size mismatch");
    if (d.sceneIdentity.empty() || d.buildIdentity.empty()) fail("Missing checkpoint identity");
    const auto& c = d.camera;
    const auto& k = d.controller;
    if (!finite(c.position) || !finite(c.lookAt) || !finite(c.view) ||
        !finite(c.up) || !finite(c.right) || !finite(c.fov) || !finite(c.pixelLength) ||
        !(c.pixelLength[0] > 0) || !(c.pixelLength[1] > 0) ||
        !finite(k.originalLookAt) || !finite(k.cameraPosition) ||
        !std::isfinite(k.zoom) || !(k.zoom > 0) ||
        !std::isfinite(k.theta) || !std::isfinite(k.phi))
        fail("Invalid checkpoint camera/controller");
    for (float x : d.accumulation)
        if (!std::isfinite(x)) fail("Nonfinite checkpoint accumulation");
}

void replaceCompleted(const std::filesystem::path& temp, const std::filesystem::path& path)
{
#ifdef _WIN32
    if (!MoveFileExW(temp.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "Replacing checkpoint");
    }
#else
    // Same-directory POSIX rename replaces an existing file without an unlink gap.
    std::filesystem::rename(temp, path);
#endif
}
} // namespace

void requireCompatible(const Data& saved, const Expected& current)
{
    if (saved.buildIdentity != current.buildIdentity)
        fail("Checkpoint rendering build/settings differ; start a fresh render");
    if (saved.sceneIdentity != current.sceneIdentity)
        fail("Checkpoint scene content differs (only target samples, FILE, and sorting may change)");
    if (saved.width != current.width || saved.height != current.height ||
        saved.traceDepth != current.traceDepth)
        fail("Checkpoint resolution or trace depth differs");
}

void write(const std::filesystem::path& path, const Data& d)
{
    validate(d);
    Writer w;
    w.bytes.reserve(d.accumulation.size() * 4 + d.sceneIdentity.size() + 256);
    w.bytes.insert(w.bytes.end(), std::begin(magic), std::end(magic));
    w.u32(kFormatVersion);
    w.u32(kRendererVersion);
    w.u32(d.width); w.u32(d.height); w.u32(d.traceDepth); w.u32(d.completedSamples);
    w.u32(d.sortMaterials ? 1u : 0u);
    w.string(d.sceneIdentity); w.string(d.buildIdentity);
    const auto& c = d.camera;
    
    w.floats(c.position); w.floats(c.lookAt); w.floats(c.view);
    w.floats(c.up); w.floats(c.right); w.floats(c.fov); w.floats(c.pixelLength);
    const auto& k = d.controller;
    w.f32(k.zoom); w.f32(k.theta); w.f32(k.phi);
    w.floats(k.originalLookAt); w.floats(k.cameraPosition);
    w.u64(d.accumulation.size());
    for (float x : d.accumulation) w.f32(x);
    w.u64(checksum(w.bytes.data(), w.bytes.size()));

    auto temp = path;
    temp += ".tmp";
    try {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        if (!f) fail("Cannot open temporary checkpoint file");
        f.write(reinterpret_cast<const char*>(w.bytes.data()),
                static_cast<std::streamsize>(w.bytes.size()));
        f.flush();
        if (!f) fail("Checkpoint write/flush failed");
        f.close();
        if (!f) fail("Checkpoint close failed");
        replaceCompleted(temp, path);
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        throw;
    }
}

Data read(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail("Cannot open checkpoint");
    const auto length = f.tellg();
    if (length < 24) fail("Truncated checkpoint");
    f.seekg(0);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) fail("Checkpoint read failed");
    if (!std::equal(std::begin(magic), std::end(magic), bytes.begin())) fail("Not a path tracer checkpoint");
    Reader r{bytes, 8};
    if (r.u32() != kFormatVersion) fail("Unsupported checkpoint format version");
    if (r.u32() != kRendererVersion) fail("Incompatible renderer/sampling version");
    Reader footer{bytes, bytes.size() - 8};
    if (footer.u64() != checksum(bytes.data(), bytes.size() - 8)) fail("Checkpoint checksum mismatch");
    Data d;
    d.width = r.u32(); d.height = r.u32(); d.traceDepth = r.u32(); d.completedSamples = r.u32();
    const auto sort = r.u32();
    if (sort > 1) fail("Invalid checkpoint sort flag");
    d.sortMaterials = sort != 0;
    d.sceneIdentity = r.string(); d.buildIdentity = r.string();
    auto& c = d.camera;
    r.floats(c.position); r.floats(c.lookAt); r.floats(c.view);
    r.floats(c.up); r.floats(c.right); r.floats(c.fov); r.floats(c.pixelLength);
    auto& k = d.controller;
    k.zoom = r.f32(); k.theta = r.f32(); k.phi = r.f32();
    r.floats(k.originalLookAt); r.floats(k.cameraPosition);
    const auto count = r.u64();
    const auto pixels = std::uint64_t(d.width) * d.height;
    if (!pixels || pixels > std::numeric_limits<int>::max() / 4 || count != pixels * 3)
        fail("Invalid checkpoint RGB length");
    r.need(8); // Check footer space before subtraction.
    if (count != (bytes.size() - r.pos - 8) / 4 ||
        (bytes.size() - r.pos - 8) % 4 != 0) fail("Checkpoint payload length mismatch");
    d.accumulation.resize(static_cast<std::size_t>(count));
    for (float& x : d.accumulation) x = r.f32();
    validate(d);
    return d;
}
} // namespace checkpoint
