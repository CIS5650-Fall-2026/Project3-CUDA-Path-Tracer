#include "scene.h"
#include "mesh.h"
#include "bvh.h"
#include "hdri.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <stb_image.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

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

int Scene::loadTexture(const std::string& path, bool srgb)
{
    // create key so we can reuse files for same use cases among materials/objects
    // saves loading time and resource usage
    std::string key = path + (srgb ? "|srgb" : "|linear");
    auto it = textureCache.find(key);
    if (it != textureCache.end()) return it->second;

    // Texture I/O
    // push back TextureData to scene textures
    int w, h, channels;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data)
    {
        cout << "error! Couldn't load texture " << path << ": " << stbi_failure_reason() << endl;
        return -1;
    }
    TextureData tex;
    tex.width = w;
    tex.height = h;
    tex.srgb = srgb;
    tex.rgba.assign(data, data + (size_t)w * h * 4);
    stbi_image_free(data);

    textures.push_back(std::move(tex));
    textureCache[key] = (int)textures.size() - 1;
    cout << "Loaded texture " << path << " (" << w << "x" << h << ")" << endl;
    return textureCache[key];
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = p.value("REFLECTIVE", 1.0f);
            newMaterial.hasRefractive = p.value("REFRACTIVE", 0.0f);
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
            newMaterial.roughness = glm::clamp(p.value("ROUGHNESS", 0.0f), 0.0f, 1.0f);
        }

        // Find color or bump textures, and keep track of parameters and if they were there or not
        newMaterial.colorTex = -1;
        newMaterial.bumpTex = -1;
        newMaterial.texScale = p.value("TEXTURE_SCALE", 1.0f);
        newMaterial.bumpStrength = p.value("BUMP_STRENGTH", 0.01f);
        newMaterial.triplanar = p.value("MAPPING", std::string("uv")) == "triplanar";
        if (p.contains("TEXTURE"))
        {
            fs::path texPath = fs::path(jsonName).parent_path() / p["TEXTURE"].get<std::string>();
            newMaterial.colorTex = loadTexture(texPath.string(), true);
        }
        if (p.contains("BUMP_MAP"))
        {
            fs::path bumpPath = fs::path(jsonName).parent_path() / p["BUMP_MAP"].get<std::string>();
            newMaterial.bumpTex = loadTexture(bumpPath.string(), false);
            if (newMaterial.bumpTex >= 0)
            {
                const TextureData& bump = textures[newMaterial.bumpTex];
                newMaterial.bumpTexelSize = glm::vec2(1.0f / bump.width, 1.0f / bump.height);
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.hasUVs = false;
        newGeom.visibleInGlass = p.value("VISIBLE_IN_GLASS", true);
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            // initializing and filling a temporary triangle array
            std::vector<Triangle> newTriangles;
            std::string fileName = p["FILENAME"];
            // mesh paths are relative to the scene JSON's own directory, not the working directory
            fs::path meshPath = fs::path(jsonName).parent_path() / fileName;
            if (!loadMeshTriangles(meshPath.string(), newTriangles, newGeom.hasUVs)) {
                std::cout << "error! Failed to load " << meshPath << std::endl;
                std::cout << "mesh will not render" << std::endl;
            }

            // position we will add upcoming bvh node to
            size_t nodesBefore = bvhNodes.size();

            // construct bvh tree
            newGeom.bvhRoot = constructBVH(newTriangles, bvhNodes);

            // starting index for this mesh's triangles is the end of the scene's current triangles
            newGeom.triangleStart = triangles.size();
            newGeom.triangleCount = newTriangles.size();

            // Shift all bvh triangles up by triangle start, since we can have more than one bvh
            for (size_t nodeIdx = nodesBefore; nodeIdx < bvhNodes.size(); ++nodeIdx)
                if (bvhNodes[nodeIdx].triangleCount > 0)
                    bvhNodes[nodeIdx].triangleStart += newGeom.triangleStart;

            // adding to scene's triangles
            triangles.insert(triangles.end(), newTriangles.begin(), newTriangles.end());

            // finally, we mark the geometry as a mesh
            newGeom.type = MESH;
        }
        else
        {
            std::cout << "error! Unidentified geometry type in " << jsonName << ": " << type << std::endl;
            std::cout << "rendering as sphere" << std::endl;
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
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

    // record which geoms are lights for NEE
    for (int i = 0; i < (int)geoms.size(); ++i)
    {
        if (materials[geoms[i].materialid].emittance > 0.0f)
        {
            lightGeomIndices.push_back(i);
        }
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];

    if (cameraData.contains("HDRI"))
    {
        fs::path hdriPath = fs::path(jsonName).parent_path() / cameraData["HDRI"].get<std::string>();
        hdriIntensity = cameraData.value("HDRI_INTENSITY", 1.0f);
        hdriRotation = cameraData.value("HDRI_ROTATION", 0.0f) / 360.0f;
        cout << "Loading HDRI " << hdriPath.string() << " ..." << endl;
        if (!loadHDRI(hdriPath.string(), hdriPixels, hdriWidth, hdriHeight))
        {
            cout << "HDRI will not render, falling back to BACKGROUND_COLOR" << endl;
            hdriPixels.clear();
            hdriWidth = hdriHeight = 0;
        }
    }
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

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // 0 = no blur
    float fstop = cameraData.value("FSTOP", 0.0f);
    // height of fake camera sensor in mm
    float sensorHeightMm = cameraData.value("SENSOR_HEIGHT", 24.0f);
    // scene meters per unit
    float metersPerUnit = cameraData.value("METERS_PER_UNIT", 1.0f);
    // distance from fake camera sensor and focus plane
    camera.focalDistance = cameraData.value("FOCUS_DISTANCE", glm::length(camera.lookAt - camera.position));
    float focalLengthM = (sensorHeightMm * 0.001f * 0.5f) / yscaled;
    camera.focalLengthMm = focalLengthM * 1000.0f;
    camera.focalLength = focalLengthM / metersPerUnit;
    camera.fstop = fstop;
    camera.lensRadius = 0.0f;
    if (fstop > 0.0f)
    {
        camera.lensRadius = camera.focalLength / (2.0f * fstop);
        cout << "DOF: " << camera.focalLengthMm << "mm lens at f/" << fstop << ", aperture radius "
             << camera.lensRadius << " units, focused at " << camera.focalDistance << " units" << endl;
    }

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
