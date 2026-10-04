#include "scene.h"

#include "utilities.h"
#include "gltf_loader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>

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

static int buildBVH(std::vector<Triangle>& tris, std::vector<BVHNode>& nodes,
    int start, int count)
{
    int idx = (int)nodes.size();
    nodes.emplace_back();
    AABB b, cb;
    for (int i = start; i < start + count; ++i) {
        b.grow(tris[i].aabb);
        cb.grow((tris[i].aabb.minBound + tris[i].aabb.maxBound) * 0.5f);
    }
    nodes[idx].bounds = b;

    if (count <= 4) {
        nodes[idx].firstTri = start;
        nodes[idx].triCount = count;
        return idx;
    }
    glm::vec3 ext = cb.maxBound - cb.minBound;
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    int mid = start + count / 2;
    std::nth_element(tris.begin() + start, tris.begin() + mid, tris.begin() + start + count,
        [axis](const Triangle& a, const Triangle& c) {
            return (a.aabb.minBound[axis] + a.aabb.maxBound[axis]) <
                (c.aabb.minBound[axis] + c.aabb.maxBound[axis]);
        });
    int l = buildBVH(tris, nodes, start, mid - start);
    int r = buildBVH(tris, nodes, mid, start + count - mid);
    nodes[idx].left = l;   // assign after recursion: emplace_back may reallocate
    nodes[idx].right = r;
    return idx;
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
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = p.value("REFLECTIVITY", 1.0f);
            newMaterial.specular.exponent = p.value("EXPONENT", 0.0f);   // 0 = perfect mirror
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // --- OBJECTS PARSING ---
    const auto& objectsData = data["Objects"];

    // compute JSON parent directory once so relative FILE paths resolve to the JSON's folder
    std::filesystem::path jsonParent = std::filesystem::path(jsonName).parent_path();

    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

        Geom newGeom;

        // Parse transformations shared by all object types
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

        if (type == "mesh" || type == "gltf")
        {
            std::string filename = p["FILE"];
            // If the FILE path is relative, resolve it relative to the JSON file location.
            std::filesystem::path filePath(filename);
            if (filePath.is_relative())
            {
                filename = (jsonParent / filePath).string();
            }

            newGeom.type = MESH;
            newGeom.materialid = MatNameToID.count(p["MATERIAL"]) ? MatNameToID[p["MATERIAL"]] : 0;

            // loadGLTF updates newGeom.triangleStartIdx, triangleCount, and boundingBox
            if (loadGLTF(filename, this->triangles, newGeom))
            {
                newGeom.bvhRoot = buildBVH(this->triangles, this->bvhNodes, newGeom.triangleStartIdx, newGeom.triangleCount);
                this->meshAABB.grow(newGeom.boundingBox);
                this->geoms.push_back(newGeom);
                std::cout << "Loaded triangles count: " << newGeom.triangleCount << std::endl;
                std::cout << "BVH nodes: " << this->bvhNodes.size() << ", root: " << newGeom.bvhRoot << std::endl;
            }
            else
            {
                std::cerr << "Failed to load glTF file: " << filename << std::endl;
            }
        }
        else
        {
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else
            {
                newGeom.type = SPHERE;
            }
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            this->geoms.push_back(newGeom);
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

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
