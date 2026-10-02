#include "checkpoint.h"
#include "json.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
using json = nlohmann::json;
namespace
{
uint64_t hashBytes(const void *memory, size_t size, uint64_t hash = 14695981039346656037ull)
{
    const auto *bytes = static_cast<const unsigned char *>(memory);
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}
std::string signature(const Scene &scene)
{
    const auto &o = scene.state.options;
    json options = {o.materialSorting, o.antialiasing,  o.russianRoulette, o.refraction, o.depthOfField,
                    o.directLighting,  o.halton,        o.motionBlur,      o.compaction, o.sharedCompaction,
                    o.meshCulling,     o.postprocess,   o.rouletteDepth,   o.seed,       o.exposure,
                    o.gamma,           o.bloomStrength, o.bloomThreshold,  o.bloomRadius};
    auto text = options.dump();
    return std::to_string(
        hashBytes(text.data(), text.size(), hashBytes(scene.sourceSignature.data(), scene.sourceSignature.size())));
}
json vectorJson(glm::vec3 v)
{
    return {v.x, v.y, v.z};
}
glm::vec3 readVector(const json &v)
{
    if (!v.is_array() || v.size() != 3)
        throw std::runtime_error("Invalid checkpoint camera");
    glm::vec3 r(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>());
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(r[k]))
            throw std::runtime_error("Non-finite checkpoint camera");
    return r;
}
} // namespace
void saveCheckpoint(const Scene &scene, int samples, const std::string &path)
{
    if (samples <= 0)
        throw std::runtime_error("No samples to checkpoint");
    const auto &c = scene.state.camera;
    json meta = {{"version", 1},
                 {"signature", signature(scene)},
                 {"samples", samples},
                 {"width", c.resolution.x},
                 {"height", c.resolution.y},
                 {"position", vectorJson(c.position)},
                 {"lookAt", vectorJson(c.lookAt)},
                 {"view", vectorJson(c.view)},
                 {"up", vectorJson(c.up)},
                 {"right", vectorJson(c.right)}};
    std::string metadata = meta.dump();
    uint32_t length = static_cast<uint32_t>(metadata.size());
    std::vector<float> pixels;
    pixels.reserve(scene.state.image.size() * 3);
    for (auto p : scene.state.image)
        for (int k = 0; k < 3; ++k)
            pixels.push_back(p[k]);
    uint64_t checksum =
        hashBytes(pixels.data(), pixels.size() * sizeof(float), hashBytes(metadata.data(), metadata.size()));
    std::filesystem::path destination(path), temporary(path + ".tmp");
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write("CUDAPT01", 8);
    out.write(reinterpret_cast<const char *>(&length), sizeof(length));
    out.write(metadata.data(), metadata.size());
    out.write(reinterpret_cast<const char *>(pixels.data()), pixels.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));
    out.flush();
    if (!out)
        throw std::runtime_error("Cannot write checkpoint: " + path);
    out.close();
    if (!out)
        throw std::runtime_error("Cannot close checkpoint: " + path);
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot replace checkpoint: " + path);
#else
    std::filesystem::rename(temporary, destination);
#endif
}
int loadCheckpoint(Scene &scene, const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open checkpoint: " + path);
    char magic[8]{};
    uint32_t length = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char *>(&length), sizeof(length));
    if (!in || std::memcmp(magic, "CUDAPT01", 8) || length > 1024 * 1024)
        throw std::runtime_error("Invalid checkpoint header");
    std::string metadata(length, '\0');
    in.read(&metadata[0], length);
    std::vector<float> pixels(scene.state.image.size() * 3);
    in.read(reinterpret_cast<char *>(pixels.data()), pixels.size() * sizeof(float));
    uint64_t stored = 0;
    in.read(reinterpret_cast<char *>(&stored), sizeof(stored));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Checkpoint size mismatch or truncated file");
    uint64_t actual =
        hashBytes(pixels.data(), pixels.size() * sizeof(float), hashBytes(metadata.data(), metadata.size()));
    if (stored != actual)
        throw std::runtime_error("Checkpoint checksum mismatch");
    json meta = json::parse(metadata);
    auto camera = scene.state.camera;
    if (meta.at("version").get<int>() != 1 || meta.at("signature").get<std::string>() != signature(scene) ||
        meta.at("width").get<int>() != camera.resolution.x || meta.at("height").get<int>() != camera.resolution.y)
        throw std::runtime_error("Checkpoint scene/settings mismatch");
    int samples = meta.at("samples").get<int>();
    if (samples <= 0)
        throw std::runtime_error("Invalid checkpoint sample count");
    camera.position = readVector(meta.at("position"));
    camera.lookAt = readVector(meta.at("lookAt"));
    camera.view = readVector(meta.at("view"));
    camera.up = readVector(meta.at("up"));
    camera.right = readVector(meta.at("right"));
    if (fabs(glm::length(camera.view) - 1) > 1e-4f || fabs(glm::length(camera.up) - 1) > 1e-4f ||
        fabs(glm::length(camera.right) - 1) > 1e-4f || fabs(glm::dot(camera.view, camera.up)) > 1e-4f ||
        fabs(glm::dot(camera.view, camera.right)) > 1e-4f || fabs(glm::dot(camera.right, camera.up)) > 1e-4f)
        throw std::runtime_error("Invalid checkpoint camera basis");
    for (float v : pixels)
        if (!std::isfinite(v) || v < 0)
            throw std::runtime_error("Invalid checkpoint radiance");
    for (size_t i = 0; i < scene.state.image.size(); ++i)
        scene.state.image[i] = glm::vec3(pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]);
    scene.state.camera = camera;
    return samples;
}
