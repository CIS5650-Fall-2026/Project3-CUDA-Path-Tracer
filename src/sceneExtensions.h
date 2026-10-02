#pragma once
#include "scene.h"
#include "json.hpp"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace sceneExtensions
{
using json = nlohmann::json;
inline glm::vec3 vec(const json &v)
{
    if (!v.is_array() || v.size() != 3)
        throw std::runtime_error("Expected a three-component vector");
    glm::vec3 result(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>());
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(result[i]))
            throw std::runtime_error("Non-finite scene value");
    return result;
}

inline void validateMaterial(const Material& m, const json& p)
{
    vec(p.at("RGB"));
    if (!std::isfinite(m.emittance) || m.emittance < 0 ||
        !std::isfinite(m.indexOfRefraction) || m.indexOfRefraction <= 0)
        throw std::runtime_error("Invalid emittance or IOR");
    for (int k = 0; k < 3; ++k)
        if (m.color[k] < 0 || (p.at("TYPE") != "Emitting" && m.color[k] > 1))
            throw std::runtime_error("Invalid material RGB");
}
inline void validateGeometry(const Geom& g, const json& p)
{
    vec(p.at("TRANS")); vec(p.at("ROTAT")); vec(p.at("SCALE"));
    for (int k = 0; k < 3; ++k)
        if (g.scale[k] <= 0) throw std::runtime_error("Object scale must be positive");
}
inline void cameraSettings(Camera& cam, const json& c, const RenderState& state, float fovy)
{
    if (cam.resolution.x <= 0 || cam.resolution.y <= 0 || cam.resolution.x > 16384 ||
        cam.resolution.y > 16384 || state.iterations == 0 || state.iterations > 2147483647u ||
        state.traceDepth < 1 || state.traceDepth > 64)
        throw std::runtime_error("Invalid resolution, iterations or depth (1..64)");
    vec(c.at("EYE")); vec(c.at("LOOKAT")); vec(c.at("UP"));
    if (glm::length(cam.lookAt - cam.position) < 1e-6f ||
        glm::length(glm::cross(cam.lookAt - cam.position, cam.up)) < 1e-6f)
        throw std::runtime_error("Degenerate camera basis");
    if (!std::isfinite(fovy) || fovy <= 0 || fovy >= 89)
        throw std::runtime_error("FOVY must be between 0 and 89 degrees");
    cam.apertureRadius = c.value("APERTURE", 0.0f);
    cam.focalDistance = c.value("FOCAL_DISTANCE", glm::length(cam.lookAt - cam.position));
    cam.shutterOpen = c.value("SHUTTER_OPEN", 0.0f);
    cam.shutterClose = c.value("SHUTTER_CLOSE", 0.0f);
    if (!std::isfinite(cam.apertureRadius) || !std::isfinite(cam.focalDistance) || !std::isfinite(cam.shutterOpen) ||
        !std::isfinite(cam.shutterClose) || cam.apertureRadius < 0 || cam.focalDistance <= 0 ||
        cam.shutterClose < cam.shutterOpen)
        throw std::runtime_error("Invalid lens or shutter settings");
}
inline void finish(Scene& scene, json data)
{
    auto& state = scene.state;
    const auto& triangles = scene.triangles;
    auto& sourceSignature = scene.sourceSignature;
    auto r = data.value("Renderer", json::object());
    auto &o = state.options;
    o.materialSorting = r.value("MATERIAL_SORTING", true);
    o.antialiasing = r.value("ANTIALIASING", true);
    o.russianRoulette = r.value("RUSSIAN_ROULETTE", true);
    o.rouletteDepth = r.value("RR_DEPTH", 3);
    o.refraction = r.value("REFRACTION", true);
    o.depthOfField = r.value("DEPTH_OF_FIELD", true);
    o.directLighting = r.value("DIRECT_LIGHTING", true);
    o.motionBlur = r.value("MOTION_BLUR", true);
    o.compaction = r.value("COMPACTION", true);
    o.sharedCompaction = r.value("SHARED_COMPACTION", true);
    o.meshCulling = r.value("MESH_CULLING", true);
    o.seed = r.value("SEED", 1u);
    std::string sampler = r.value("SAMPLER", std::string("halton"));
    if (sampler != "halton" && sampler != "random")
        throw std::runtime_error("SAMPLER must be halton or random");
    o.halton = sampler == "halton";
    o.postprocess = r.value("POSTPROCESS", false);
    o.exposure = r.value("EXPOSURE", 0.0f);
    o.gamma = r.value("GAMMA", 2.2f);
    o.bloomStrength = r.value("BLOOM_STRENGTH", 0.15f);
    o.bloomThreshold = r.value("BLOOM_THRESHOLD", 1.0f);
    o.bloomRadius = r.value("BLOOM_RADIUS", 6);
    if (o.rouletteDepth < 1 || o.bloomRadius < 0 || o.bloomRadius > 32 || !std::isfinite(o.gamma) || o.gamma <= 0 ||
        !std::isfinite(o.exposure) || fabs(o.exposure) > 20 || !std::isfinite(o.bloomStrength) || o.bloomStrength < 0 ||
        !std::isfinite(o.bloomThreshold) || o.bloomThreshold < 0)
        throw std::runtime_error("Invalid roulette or postprocess settings");
    state.displayImage = state.image;
    data["Camera"].erase("ITERATIONS");
    data["Camera"].erase("FILE");
    sourceSignature = data.dump();
    for (const auto &t : triangles)
        for (auto v : {t.a, t.b, t.c})
            for (int k = 0; k < 3; ++k)
                sourceSignature.append(reinterpret_cast<const char *>(&v[k]), sizeof(float));
}
} // namespace sceneExtensions
