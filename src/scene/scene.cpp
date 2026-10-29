#include "scene/scene.h"

#include "scene/gltf_loader.h"
#include "timing.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include "json.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace
{
// Defaults for a glTF scene, matching the scene JSONs in scenes/.
const int DEFAULT_HEIGHT = 1024;
const int DEFAULT_ITERATIONS = 5000;
const int DEFAULT_TRACE_DEPTH = 8;
// Degrees, vertical.
const float DEFAULT_FOVY = 45.0f;

// Throws rather than fatal() so the window can show the message and keep its scene.
[[noreturn]] void sceneError(const char* format, ...)
{
    char message[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof message, format, args);
    va_end(args);
    throw std::runtime_error(message);
}

RenderSettings withOverrides(RenderSettings settings, const SceneOverrides& ov)
{
    if (ov.width > 0 && ov.height > 0)
    {
        settings.resolution = glm::ivec2(ov.width, ov.height);
    }
    if (ov.iterations > 0)
    {
        settings.iterations = ov.iterations;
    }
    if (ov.traceDepth > 0)
    {
        settings.traceDepth = ov.traceDepth;
    }
    return settings;
}

void setCameraType(CameraPose& pose, const std::string& value)
{
    pose.lensFile.clear();
    if (value == "pinhole")
    {
        pose.type = CAMERA_PINHOLE;
    }
    else if (value == "thin")
    {
        pose.type = CAMERA_THIN_LENS;
    }
    else
    {
        pose.type = CAMERA_REAL_LENS;
        pose.lensFile = value;
    }
}

CameraPose withOverrides(CameraPose pose, const SceneOverrides& ov)
{
    if (!ov.lens.empty())
    {
        setCameraType(pose, ov.lens);
    }
    if (ov.apertureRadius > 0.0f)
    {
        pose.apertureRadius = ov.apertureRadius;
    }
    if (ov.focusDistance > 0.0f)
    {
        pose.focusDistance = ov.focusDistance;
    }
    if (ov.fovy > 0.0f)
    {
        pose.fovy = ov.fovy;
    }
    if (ov.unitsPerMeter > 0.0f)
    {
        pose.unitsPerMeter = ov.unitsPerMeter;
    }
    return pose;
}

// Throws, like json::at, when the key is missing or not three numbers.
glm::vec3 vec3At(const json& object, const char* key)
{
    const json& v = object.at(key);
    if (!v.is_array() || v.size() != 3)
    {
        throw std::runtime_error(std::string(key) + " is not an array of three numbers");
    }
    return glm::vec3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
}

const std::string SCENE_DIR = "scenes/";
const std::string CATALOG = "scenes/catalog.json";

// (name, path relative to scenes/) pairs in file order.
std::vector<std::pair<std::string, std::string>> readCatalog()
{
    std::vector<std::pair<std::string, std::string>> entries;
    std::ifstream f(CATALOG);
    if (!f)
    {
        return entries;
    }
    try
    {
        const json catalog = json::parse(f);
        for (const auto& item : catalog.items())
        {
            entries.emplace_back(item.key(), item.value().get<std::string>());
        }
    }
    catch (const std::exception& e)
    {
        sceneError("%s: %s", CATALOG.c_str(), e.what());
    }
    return entries;
}

std::vector<std::string> sceneJsonNames()
{
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(SCENE_DIR))
    {
        if (entry.is_regular_file() && lowercaseExtension(entry.path().string()) == ".json"
            && entry.path().filename() != "catalog.json")
        {
            names.push_back(entry.path().stem().string());
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}
}  // namespace

std::string findSceneFile(const std::string& argument)
{
    if (std::filesystem::is_regular_file(argument))
    {
        return argument;
    }
    const std::filesystem::path path(argument);
    if (path.has_extension() || path.has_parent_path())
    {
        sceneError("no scene file %s", argument.c_str());
    }
    const std::string sceneJson = SCENE_DIR + argument + ".json";
    if (std::filesystem::is_regular_file(sceneJson))
    {
        return sceneJson;
    }
    for (const auto& [name, file] : readCatalog())
    {
        if (name == argument)
        {
            const std::string catalogFile = SCENE_DIR + file;
            if (!std::filesystem::is_regular_file(catalogFile))
            {
                sceneError("%s is %s, which is not downloaded (see the README's asset list)", argument.c_str(),
                    catalogFile.c_str());
            }
            return catalogFile;
        }
    }
    sceneError("no scene named %s: neither %s nor an entry in %s (--list prints the names)", argument.c_str(),
        sceneJson.c_str(), CATALOG.c_str());
}

std::vector<SceneName> sceneNames()
{
    std::vector<SceneName> names;
    if (!std::filesystem::is_directory(SCENE_DIR))
    {
        return names;
    }
    for (const std::string& name : sceneJsonNames())
    {
        names.push_back({name, true});
    }
    for (const auto& [name, file] : readCatalog())
    {
        names.push_back({name, std::filesystem::is_regular_file(SCENE_DIR + file)});
    }
    return names;
}

void listScenes()
{
    if (!std::filesystem::is_directory(SCENE_DIR))
    {
        fatal("no %s folder here; run from the repository root", SCENE_DIR.c_str());
    }
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(SCENE_DIR))
    {
        if (entry.is_regular_file() && lowercaseExtension(entry.path().string()) == ".json"
            && entry.path().filename() != "catalog.json")
        {
            names.push_back(entry.path().stem().string());
        }
    }
    std::sort(names.begin(), names.end());
    printf("Scene JSONs in %s:\n", SCENE_DIR.c_str());
    for (const std::string& name : names)
    {
        printf("  %s\n", name.c_str());
    }
    printf("glTF scenes in %s:\n", CATALOG.c_str());
    for (const auto& [name, file] : readCatalog())
    {
        const bool downloaded = std::filesystem::is_regular_file(SCENE_DIR + file);
        printf("  %-20s %s%s\n", name.c_str(), file.c_str(), downloaded ? "" : "  (not downloaded)");
    }
}

Scene::Scene(string filename, const SceneOverrides& ov)
{
    TimingScope timing("load.scene");
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    const string ext = lowercaseExtension(filename);
    // Every load error leaves as a runtime_error starting with the file's name.
    try
    {
        if (ext == ".json")
        {
            loadFromJSON(filename, ov);
        }
        else if (ext == ".gltf" || ext == ".glb")
        {
            loadFromGltf(filename, ov);
        }
        else
        {
            sceneError("not a scene file (.json, .gltf or .glb)");
        }
        environment = loadEnvironment(ov.environmentFile.empty()
                ? environmentSource
                : EnvironmentSource{ ov.environmentFile, glm::vec3(1.0f), 0.0f },
            ov.environmentCompensation);
        environmentLight = ov.environmentLight;
        buildLights();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(filename + ": " + e.what());
    }
}

void Scene::bounds(size_t first, size_t last, glm::vec3& lo, glm::vec3& hi) const
{
    lo = glm::vec3(FLT_MAX);
    hi = glm::vec3(-FLT_MAX);
    auto grow = [&](const Geom& geom, const glm::vec3& p) {
        const glm::vec3 world = glm::vec3(geom.transform * glm::vec4(p, 1.0f));
        lo = glm::min(lo, world);
        hi = glm::max(hi, world);
    };
    for (size_t g = first; g < last; ++g)
    {
        const Geom& geom = geoms[g];
        if (geom.type == GeomType::MESH)
        {
            const TriangleMesh& mesh = meshes[geom.meshId];
            for (int t = 0; t < mesh.triCount; ++t)
            {
                const glm::ivec3 tri = indices[mesh.indexOffset + t];
                for (int k = 0; k < 3; ++k)
                {
                    grow(geom, positions[tri[k]]);
                }
            }
        }
        else
        {
            // A sphere has radius 0.5 and a cube half-width 0.5 before the transform.
            for (int c = 0; c < 8; ++c)
            {
                grow(geom, glm::vec3(c & 1 ? 0.5f : -0.5f, c & 2 ? 0.5f : -0.5f, c & 4 ? 0.5f : -0.5f));
            }
        }
    }
}

void Scene::loadFromJSON(const std::string& jsonName, const SceneOverrides& ov)
{
    std::ifstream f(jsonName);
    if (!f)
    {
        sceneError("cannot open");
    }
    const json data = json::parse(f);
    // Each material TYPE maps onto Material, starting from glTF's default material.
    std::unordered_map<std::string, uint32_t> materialIdsByName;
    for (const auto& item : data.at("Materials").items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        const glm::vec3 rgb = vec3At(p, "RGB");
        const std::string type = p.at("TYPE");
        Material newMaterial{};
        if (type == "Diffuse")
        {
            newMaterial.baseColor = rgb;
            newMaterial.metallic = 0.0f;
            newMaterial.specularFactor = 0.0f;
        }
        else if (type == "Emitting")
        {
            newMaterial.baseColor = glm::vec3(0.0f);
            newMaterial.metallic = 0.0f;
            newMaterial.specularFactor = 0.0f;
            newMaterial.emission = rgb * p.at("EMITTANCE").get<float>();
        }
        else if (type == "Specular")
        {
            newMaterial.baseColor = rgb;
            newMaterial.metallic = 1.0f;
            newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
        }
        else if (type == "Refractive")
        {
            newMaterial.baseColor = rgb;
            newMaterial.metallic = 0.0f;
            newMaterial.roughness = 0.0f;
            newMaterial.transmission = 1.0f;
            newMaterial.ior = p.at("IOR").get<float>();
        }
        else
        {
            sceneError("material %s has an unknown TYPE \"%s\" (Diffuse, Emitting, Specular or Refractive)",
                name.c_str(), type.c_str());
        }
        materialIdsByName[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // Not operator[]: it would map a typo to 0, the light in every Cornell scene.
    auto materialIndex = [&](const std::string& name) -> int {
        auto found = materialIdsByName.find(name);
        if (found == materialIdsByName.end())
        {
            sceneError("unknown material %s", name.c_str());
        }
        return (int)found->second;
    };

    // FILE paths are relative to the scene file's folder.
    const std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);

    // ROTATION is degrees about +Y, the same number as a Blender Mapping node's Z rotation.
    if (data.contains("Environment"))
    {
        const json& e = data.at("Environment");
        const float strength = e.value("STRENGTH", 1.0f);
        if (e.contains("FILE"))
        {
            environmentSource.file = sceneDir + e.at("FILE").get<std::string>();
            environmentSource.radiance = glm::vec3(strength);
        }
        else
        {
            environmentSource.radiance = vec3At(e, "RGB") * strength;
        }
        environmentSource.rotation = glm::radians(e.value("ROTATION", 0.0f));
    }

    for (const auto& p : data.at("Objects"))
    {
        const std::string type = p.at("TYPE");
        const glm::mat4 transform =
            buildTransformationMatrix(vec3At(p, "TRANS"), vec3At(p, "ROTAT"), vec3At(p, "SCALE"));

        if (type == "mesh")
        {
            // MATERIAL, when given, replaces the file's materials.
            const int materialOverride = p.contains("MATERIAL") ? materialIndex(p.at("MATERIAL").get<std::string>()) : -1;
            const std::string file = sceneDir + p.at("FILE").get<std::string>();
            if (!loadGltf(file, transform, materialOverride, *this))
            {
                sceneError("cannot load %s", file.c_str());
            }
            continue;
        }

        Geom newGeom{};
        if (type == "cube")
        {
            newGeom.type = GeomType::CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = GeomType::SPHERE;
        }
        else
        {
            sceneError("unknown object TYPE \"%s\" (cube, sphere or mesh)", type.c_str());
        }
        newGeom.materialId = materialIndex(p.at("MATERIAL").get<std::string>());
        newGeom.meshId = -1;
        newGeom.transform = transform;
        newGeom.inverseTransform = glm::inverse(transform);
        newGeom.invTranspose = glm::inverseTranspose(transform);

        geoms.push_back(newGeom);
    }
    const json& cameraData = data.at("Camera");
    const json& res = cameraData.at("RES");
    RenderSettings settings;
    settings.resolution = glm::ivec2(res.at(0).get<int>(), res.at(1).get<int>());
    settings.iterations = cameraData.at("ITERATIONS").get<int>();
    settings.traceDepth = cameraData.at("DEPTH").get<int>();
    settings.imageName = cameraData.at("FILE").get<std::string>();

    CameraPose pose;
    pose.eye = vec3At(cameraData, "EYE");
    pose.lookAt = vec3At(cameraData, "LOOKAT");
    pose.up = vec3At(cameraData, "UP");
    pose.fovy = cameraData.at("FOVY").get<float>();
    pose.mirrored = false;
    if (cameraData.contains("LENS"))
    {
        setCameraType(pose, cameraData.at("LENS").get<std::string>());
    }
    pose.apertureRadius = cameraData.value("APERTURE", 0.0f);
    pose.focusDistance = cameraData.value("FOCUS", 0.0f);
    // 1 reads the scene in meters, 100 in centimeters.
    pose.unitsPerMeter = data.value("UNITS_PER_METER", 1.0f);
    if (!(pose.unitsPerMeter > 0.0f))
    {
        sceneError("UNITS_PER_METER must be positive");
    }

    initRenderState(withOverrides(settings, ov), withOverrides(pose, ov));
}

void Scene::loadFromGltf(const std::string& gltfName, const SceneOverrides& ov)
{
    GltfInfo info;
    if (!loadGltf(gltfName, glm::mat4(1.0f), -1, *this, &info))
    {
        sceneError("cannot load (the loader printed why)");
    }
    if (geoms.empty())
    {
        sceneError("nothing to render");
    }

    if (info.environment)
    {
        const GltfEnvironment& e = *info.environment;
        if (e.latlongFile.empty())
        {
            environmentSource.radiance = e.radiance;
        }
        else if (std::filesystem::is_regular_file(e.latlongFile))
        {
            environmentSource.file = e.latlongFile;
            environmentSource.radiance = e.radiance;
        }
        else if (std::filesystem::is_regular_file(e.pbrtFile))
        {
            cerr << "No " << e.latlongFile << ": run python tools/convert_envmaps.py to make it from "
                 << e.pbrtFile << "; no environment until then" << endl;
        }
        else
        {
            cerr << "The PBRT environment map " << e.pbrtFile << " is not in the download; no environment" << endl;
        }
    }

    if (maxComponent(environmentSource.radiance) <= 0.0f && ov.environmentFile.empty() && punctualLights.empty()
        && std::none_of(materials.begin(), materials.end(), [](const Material& m) {
               return maxComponent(m.emission) > 0.0f;
           }))
    {
        cerr << "No light, emissive material or environment in " << gltfName << ": the render will be black" << endl;
    }

    const GltfRenderHints& hints = info.render;
    RenderSettings settings;
    settings.resolution = glm::ivec2(DEFAULT_HEIGHT);
    if (hints.width > 0 && hints.height > 0)
    {
        settings.resolution = glm::ivec2(hints.width, hints.height);
    }
    else if (info.camera && info.camera->aspectRatio > 0.0f)
    {
        settings.resolution.x = glm::max(1, (int)std::round(DEFAULT_HEIGHT * info.camera->aspectRatio));
    }
    settings.iterations = DEFAULT_ITERATIONS;
    // PBRT's max_depth counts bounces; hitting a light takes one more ray here.
    settings.traceDepth = hints.maxDepth > 0 ? hints.maxDepth + 1 : DEFAULT_TRACE_DEPTH;
    settings.imageName = std::filesystem::path(gltfName).stem().string();
    settings = withOverrides(settings, ov);

    const glm::vec3 center = 0.5f * (info.boundsMin + info.boundsMax);
    const float radius = glm::max(0.5f * glm::length(info.boundsMax - info.boundsMin), 1e-3f);

    CameraPose pose;
    if (info.camera)
    {
        const GltfCamera& c = *info.camera;
        pose.eye = c.position;
        pose.up = c.up;
        pose.fovy = glm::degrees(c.yfov);
        pose.mirrored = c.mirrored;

        // The orbit point: on the view axis nearest the scene's center, else one radius ahead.
        float ahead = glm::dot(center - c.position, c.view);
        if (ahead < 0.01f * radius)
        {
            ahead = radius;
        }
        pose.lookAt = c.position + ahead * c.view;
    }
    else
    {
        // From +Z, far enough back that the bounding sphere fits the narrower side.
        const float halfY = glm::radians(0.5f * DEFAULT_FOVY);
        const float aspect = (float)settings.resolution.x / (float)settings.resolution.y;
        const float halfAngle = glm::min(halfY, std::atan(std::tan(halfY) * aspect));
        pose.eye = center + glm::vec3(0.0f, 0.0f, radius / std::sin(halfAngle));
        pose.lookAt = center;
        pose.up = glm::vec3(0.0f, 1.0f, 0.0f);
        pose.fovy = DEFAULT_FOVY;
        pose.mirrored = false;
    }

    initRenderState(settings, withOverrides(pose, ov));
}

void Scene::initRenderState(const RenderSettings& settings, const CameraPose& pose)
{
    Camera& camera = state.camera;
    camera.resolution = settings.resolution;
    state.iterations = settings.iterations;
    state.traceDepth = settings.traceDepth;
    state.imageName = settings.imageName;

    camera.position = pose.eye;
    camera.lookAt = pose.lookAt;
    camera.up = pose.up;
    camera.mirrored = pose.mirrored;
    camera.type = pose.type;
    camera.apertureRadius = pose.apertureRadius;
    camera.focusDistance = pose.focusDistance > 0.0f ? pose.focusDistance : glm::length(pose.lookAt - pose.eye);
    camera.unitsPerMeter = pose.unitsPerMeter;
    if (camera.type == CAMERA_REAL_LENS)
    {
        lens = loadLens(pose.lensFile);  // throws with the file's name
        const LensSystem::Paraxial& p = lens.paraxial;
        printf("lens %s: %d surfaces, stop at surface %d; traced f = %.2f mm, f/%.2f, film %.3f mm behind the "
            "front vertex, entrance pupil %.2f mm behind it with radius %.2f mm\n",
            lens.name.c_str(), (int)lens.surfaces.size(), lens.stopIndex + 1, p.focalLength, p.fNumber,
            p.filmDistance, p.entrancePupilZ, p.entrancePupilRadius);
        int aspheric = 0;
        for (const LensSurface& s : lens.surfaces)
        {
            aspheric += s.asphericCount > 0 || s.conic != 0.0f;
        }
        if (aspheric > 0)
        {
            fprintf(stderr, "%d aspheric surfaces are traced as spheres of their paraxial radius for now\n", aspheric);
        }
    }

    float yscaled = tan(0.5f * pose.fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    if (camera.mirrored)
    {
        camera.right = -camera.right;
    }

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
