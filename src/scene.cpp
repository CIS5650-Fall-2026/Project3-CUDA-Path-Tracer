#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace {

    // An absent object behaves like an empty object, so all defaults remain.
    const json& optionalObject(const json& parent, const char* key)
    {
        static const json empty = json::object();
        if (!parent.contains(key)) {
            return empty;
        }
        const auto& value = parent.at(key);
        if (!value.is_object()) {
            throw std::runtime_error(std::string(key) + " must be an object");
        }
        return value;
    }

    float readFloat(const json& object, const char* key, float fallback,
        float minimum = 0.0f, float maximum = 1.0f)
    {
        if (!object.contains(key)) {
            return fallback;
        }
        const auto& value = object.at(key);
        if (!value.is_number()) {
            throw std::runtime_error(std::string(key) + " must be a number");
        }
        const float result = value.get<float>();
        if (!std::isfinite(result) || result < minimum || result > maximum) {
            throw std::runtime_error(std::string(key) + " is outside the supported range");
        }
        return result;
    }

    glm::vec3 readColor(const json& object, const char* key,
        const glm::vec3& fallback, std::size_t components)
    {
        if (!object.contains(key)) {
            return fallback;
        }
        const auto& value = object.at(key);
        if (!value.is_array() || value.size() != components) {
            throw std::runtime_error(std::string(key) + " has the wrong color-array length");
        }
        for (const auto& component : value) {
            if (!component.is_number()) {
                throw std::runtime_error(std::string(key) + " must contain numbers");
            }
            const float x = component.get<float>();
            if (!std::isfinite(x) || x < 0.0f || x > 1.0f) {
                throw std::runtime_error(std::string(key) + " components must be in [0, 1]");
            }
        }
        // Return the RGB components
        return glm::vec3(value.at(0).get<float>(), value.at(1).get<float>(),
            value.at(2).get<float>());
    }

    // Accept either a grayscale scalar or a threee component RGB value.
    glm::vec3 readCoefficient(
        const json& object,
        const char* key,
        const glm::vec3& fallback)
    {
        if (!object.contains(key)) {
            return fallback;
        }

        if (object.at(key).is_number()) {
            return glm::vec3(readFloat(object, key, 0.0f));
        }

        return readColor(object, key, fallback, 3);
    }

    Material parseMaterial(const json& p)
    {
        if (!p.is_object()) {
            throw std::runtime_error("Each material must be an object");
        }

        if (p.contains("pbrMetallicRoughness")) {
            throw std::runtime_error(
                "Convert this material to the standard/metal scene format");
        }

        Material result{};
        const std::string type =
            p.value("type", std::string("standard"));

        if (type != "transmission" && p.contains("transmissionPresetID")) {
            throw std::runtime_error("transmissionPresetID requires type = transmission");
        }

        if (type == "standard") {
            if (p.contains("metalPresetID")) {
                throw std::runtime_error(
                    "metalPresetID requires type = metal");
            }

            result.Kd = readCoefficient(p, "Kd", result.Kd);

            result.ior = readFloat(
                p, "ior", result.ior,
                1.0f, std::numeric_limits<float>::max());

            result.diffuseSigmaDegrees = readFloat(
                p, "diffuseSigmaDegrees", result.diffuseSigmaDegrees,
                0.0f, 90.0f);
        }
        else if (type == "metal") {
            if (p.contains("Kd") ||
                p.contains("ior") ||
                p.contains("diffuseSigmaDegrees")) {
                throw std::runtime_error(
                    "Metal materials use presets, not Kd, ior, "
                    "or diffuseSigmaDegrees");
            }

            result.type = MaterialType::Metal;
            result.Kd = glm::vec3(0.0f);
            result.Ks = glm::vec3(1.0f);

            if (!p.contains("metalPresetID")) {
                throw std::runtime_error(
                    "Metal materials require metalPresetID");
            }

            const auto& preset = p.at("metalPresetID");

            constexpr int count = static_cast<int>(
                sizeof(metalIORPresets) / sizeof(metalIORPresets[0]));

            if (!preset.is_number_integer() ||
                preset < 0 ||
                preset >= count) {
                throw std::runtime_error(
                    "metalPresetID must be a valid nonnegative preset index");
            }

            result.metal.setPreset(preset.get<int>());
        }
        else if (type == "transmission") {
            if (p.contains("Kd") || p.contains("Ks") || p.contains("ior") ||
                p.contains("roughness") || p.contains("diffuseSigmaDegrees") ||
                p.contains("metalPresetID")) {
                throw std::runtime_error(
                    "Transmission uses a smooth clear-glass preset; omit Kd, Ks, "
                    "ior, roughness, diffuseSigmaDegrees, and metalPresetID");
            }
            if (!p.contains("transmissionPresetID")) {
                throw std::runtime_error("Transmission materials require transmissionPresetID");
            }
            const auto& preset = p.at("transmissionPresetID");
            constexpr int count = static_cast<int>(
                sizeof(transmissionIORPresets) / sizeof(transmissionIORPresets[0]));
            if (!preset.is_number_integer() || preset < 0 || preset >= count) {
                throw std::runtime_error(
                    "transmissionPresetID must be a valid nonnegative preset index");
            }
            result.type = MaterialType::Transmission;
            result.Kd = glm::vec3(0.0f);
            result.Ks = glm::vec3(0.0f);
            result.transmission.setPreset(preset.get<int>());
        }
        else {
            throw std::runtime_error("Unknown material type: " + type);
        }

        if (result.type != MaterialType::Transmission) {
            result.Ks = readCoefficient(p, "Ks", result.Ks);
            // Preserve  existing Standard/Metal mapping including zero.
            result.setRoughness(readFloat(p, "roughness", result.roughness));
        }

        const auto& emission = optionalObject(p, "emission");

        result.emission.color = readColor(
            emission, "color", result.emission.color, 3);

        result.emission.emittance = readFloat(
            emission, "strength", result.emission.emittance,
            0.0f, std::numeric_limits<float>::max());

        return result;
    }

} // namespace


Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    if (!f) {
        throw std::runtime_error("Could not open scene: " + jsonName);
    }
    const json data = json::parse(f);
    json checkpointData = data;
    checkpointData.at("Camera").erase("ITERATIONS");
    checkpointData.at("Camera").erase("FILE");
    checkpointData.erase("sortMaterials");
    checkpointSceneIdentity = checkpointData.dump();
    const auto& materialsData = data.at("materials");
    if (!materialsData.is_array()) {
        throw std::runtime_error("materials must be an array");
    }

    materials.clear();
    geoms.clear();
    materials.reserve(materialsData.size());
    for (std::size_t i = 0; i < materialsData.size(); ++i) {
        try {
            materials.emplace_back(parseMaterial(materialsData.at(i)));
        }
        catch (const std::exception& error) {
            throw std::runtime_error("materials[" + std::to_string(i) + "]: " + error.what());
        }
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom{};
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        const auto& materialIndex = p.at("material");
        if (!materialIndex.is_number_integer() || materialIndex < 0 ||
            materialIndex >= materials.size() ||
            materialIndex > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Object material must be a valid materials-array index");
        }
        newGeom.materialid = materialIndex.get<int>();
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    RenderState& state = this->state;
    state.sortMaterials = data.value("sortMaterials", true);
    // These flags remain in checkpointSceneIdentity, unlike sorting.
    state.compactPaths = data.value("compactPaths", true);
    state.antialiasing = data.value("antialiasing", true);
    Camera& camera = state.camera;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    // Counts and pixel indices currently pass through signed int CUDA APIs.
    if (camera.resolution.x <= 0 || camera.resolution.y <= 0 ||
        static_cast<long long>(camera.resolution.x) * camera.resolution.y >
            std::numeric_limits<int>::max() / 4 ||
        state.iterations == 0 || state.iterations > std::numeric_limits<int>::max() ||
        state.traceDepth < 0) {
        throw std::runtime_error("Invalid resolution, iteration target, or trace depth");
    }
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3(0.0f));
}
