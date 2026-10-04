#include "scene.h"

#include "utilities.h"

#include "gltfLoader.h"
#include "objLoader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <cfloat>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

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

    // Test read TODO (remove)
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
            newMaterial.hasReflective = 1;
        }
        else if (p["TYPE"] == "Glass") {
          const auto& col = p["RGB"];
          newMaterial.color = glm::vec3(col[0], col[1], col[2]);
          newMaterial.hasRefractive = 1.0;
          newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];

        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);

        newGeom.transform = utilityCore::buildTransformationMatrix(
          newGeom.translation, newGeom.rotation, newGeom.scale
        );
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (type == "cube")
        {
          newGeom.type = CUBE;
        }
        else if (type == "mesh") {
          std::filesystem::path modelPath = std::filesystem::path(jsonName).parent_path() / p["FILE"].get<std::string>();

          newGeom.type = MESH;
          newGeom.triangleStart = static_cast<int>(triangles.size());

          std::string fileExtension = modelPath.extension().string();

          if (fileExtension == ".obj") {
            loadObj(modelPath.string(), newGeom.transform, triangles);
          }
          else if (fileExtension == ".gltf" || fileExtension == ".glb") {
            loadGLTF(modelPath.string(), newGeom.transform, triangles);
          }
          else {
            throw std::runtime_error("Unsupported mesh extension '" + fileExtension + "'\n");
          }

          newGeom.triangleCount = static_cast<int>(triangles.size() - newGeom.triangleStart);

          newGeom.boundsMin = glm::vec3(FLT_MAX);
          newGeom.boundsMax = glm::vec3(-FLT_MAX);

          for (int j = 0; j < newGeom.triangleCount; j++) {
            const Triangle& tri = triangles[newGeom.triangleStart + j];

            newGeom.boundsMin = glm::min(newGeom.boundsMin, tri.v0);
            newGeom.boundsMin = glm::min(newGeom.boundsMin, tri.v1);
            newGeom.boundsMin = glm::min(newGeom.boundsMin, tri.v2);

            newGeom.boundsMax = glm::max(newGeom.boundsMax, tri.v0);
            newGeom.boundsMax = glm::max(newGeom.boundsMax, tri.v1);
            newGeom.boundsMax = glm::max(newGeom.boundsMax, tri.v2);
          }

          // Offset bounds to ensure it wraps AROUND mesh
          newGeom.boundsMin -= glm::vec3(1e-4f);
          newGeom.boundsMax += glm::vec3(1e-4f);
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];

        geoms.push_back(newGeom);
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
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // Anti Aliasing Flag
    camera.antiAliasingEnabled = cameraData.value("ANTI_ALIASING", true);

    // Depth of field - optionally provided args
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);
    //  distance between eye and lookat
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    // Calculate view first to ensure right is orthogonal to view
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
