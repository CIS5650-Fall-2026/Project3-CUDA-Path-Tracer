#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <cfloat>
#include <algorithm>

using namespace std;
using json = nlohmann::json;

namespace
{
int objIndex(const std::string& field, size_t size)
{
    int value = std::stoi(field);
    int index = value > 0 ? value - 1 : static_cast<int>(size) + value;
    if (value == 0 || index < 0 || index >= static_cast<int>(size))
        throw std::runtime_error("OBJ index out of range");
    return index;
}

void loadOBJ(const std::filesystem::path& filename, std::vector<Triangle>& triangles)
{
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open mesh: " + filename.string());
    std::vector<glm::vec3> positions, normals;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream row(line);
        std::string kind;
        row >> kind;
        if (kind == "v" || kind == "vn")
        {
            glm::vec3 value;
            if (!(row >> value.x >> value.y >> value.z))
                throw std::runtime_error("Invalid OBJ vertex or normal");
            if (kind == "v") positions.push_back(value);
            else normals.push_back(value);
        }
        else if (kind == "f")
        {
            std::vector<std::pair<int, int>> face;
            std::string token;
            while (row >> token)
            {
                if (token[0] == '#') break;
                size_t slash = token.find('/');
                int vertex = objIndex(token.substr(0, slash), positions.size());
                int normal = -1;
                if (slash != std::string::npos)
                {
                    size_t second = token.find('/', slash + 1);
                    if (second != std::string::npos && second + 1 < token.size())
                        normal = objIndex(token.substr(second + 1), normals.size());
                }
                face.emplace_back(vertex, normal);
            }
            if (face.size() < 3) throw std::runtime_error("OBJ face needs three vertices");
            // Fan triangulation supports triangular and convex polygonal faces.
            for (size_t i = 1; i + 1 < face.size(); ++i)
            {
                auto a = face[0], b = face[i], c = face[i + 1];
                Triangle t{};
                t.v0 = positions[a.first]; t.v1 = positions[b.first]; t.v2 = positions[c.first];
                glm::vec3 cross = glm::cross(t.v1 - t.v0, t.v2 - t.v0);
                if (glm::dot(cross, cross) < 1e-18f) continue;
                glm::vec3 fallback = glm::normalize(cross);
                auto normalAt = [&](int n) {
                    return n >= 0 && glm::dot(normals[n], normals[n]) > 1e-18f ?
                        glm::normalize(normals[n]) : fallback;
                };
                t.n0 = normalAt(a.second); t.n1 = normalAt(b.second); t.n2 = normalAt(c.second);
                triangles.push_back(t);
            }
        }
    }
}

int buildMeshBVH(std::vector<Triangle>& triangles, std::vector<MeshBVHNode>& nodes,
    int first, int count)
{
    int index = static_cast<int>(nodes.size());
    MeshBVHNode node{};
    node.minimum = glm::vec3(FLT_MAX); node.maximum = glm::vec3(-FLT_MAX);
    glm::vec3 centroidMin(FLT_MAX), centroidMax(-FLT_MAX);
    for (int i = first; i < first + count; ++i)
    {
        const Triangle& t = triangles[i];
        node.minimum = glm::min(node.minimum, glm::min(t.v0, glm::min(t.v1, t.v2)));
        node.maximum = glm::max(node.maximum, glm::max(t.v0, glm::max(t.v1, t.v2)));
        glm::vec3 center = (t.v0 + t.v1 + t.v2) / 3.0f;
        centroidMin = glm::min(centroidMin, center); centroidMax = glm::max(centroidMax, center);
    }
    node.minimum -= glm::vec3(1e-5f); node.maximum += glm::vec3(1e-5f);
    nodes.push_back(node);
    glm::vec3 span = centroidMax - centroidMin;
    int axis = span.y > span.x ? 1 : 0;
    if (span.z > span[axis]) axis = 2;
    if (count <= 4 || span[axis] < 1e-8f)
    {
        nodes[index].first = first;
        nodes[index].count = count;
    }
    else
    {
        int middle = first + count / 2;
        std::nth_element(triangles.begin() + first, triangles.begin() + middle,
            triangles.begin() + first + count, [axis](const Triangle& a, const Triangle& b) {
                return (a.v0[axis] + a.v1[axis] + a.v2[axis]) <
                    (b.v0[axis] + b.v1[axis] + b.v2[axis]);
            });
        buildMeshBVH(triangles, nodes, first, middle - first);
        buildMeshBVH(triangles, nodes, middle, first + count - middle);
    }
    nodes[index].escape = static_cast<int>(nodes.size());
    return index;
}
}

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
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        newMaterial.secondaryColor = newMaterial.color;
        newMaterial.texture = SOLID;
        newMaterial.textureScale = p.value("TEXTURE_SCALE", 8.0f);
        if (p.contains("RGB2"))
        {
            const auto& secondary = p["RGB2"];
            newMaterial.secondaryColor = glm::vec3(secondary[0], secondary[1], secondary[2]);
        }
        const std::string texture = p.value("TEXTURE", std::string("Solid"));
        if (texture == "Checker") newMaterial.texture = CHECKER;
        if (texture == "Marble") newMaterial.texture = MARBLE;
        if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Glass" || p["TYPE"] == "Refractive")
        {
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
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
        else if (type == "torus")
        {
            newGeom.type = TORUS;
        }
        else if (type == "woven_ring")
        {
            newGeom.type = WOVEN_RING;
        }
        else if (type == "gyroid")
        {
            newGeom.type = GYROID;
        }
        else if (type == "mesh" || type == "obj")
        {
            newGeom.type = MESH;
            std::filesystem::path meshPath(p.at("FILE").get<std::string>());
            if (meshPath.is_relative()) meshPath = std::filesystem::path(jsonName).parent_path() / meshPath;
            newGeom.triangleStart = static_cast<int>(triangles.size());
            loadOBJ(meshPath, triangles);
            newGeom.triangleCount = static_cast<int>(triangles.size()) - newGeom.triangleStart;
            if (newGeom.triangleCount == 0) throw std::runtime_error("Mesh has no valid triangles");
            newGeom.bvhRoot = buildMeshBVH(triangles, meshNodes,
                newGeom.triangleStart, newGeom.triangleCount);
            std::cout << "Loaded " << meshPath.string() << ": " << newGeom.triangleCount
                << " triangles, " << meshNodes[newGeom.bvhRoot].escape - newGeom.bvhRoot
                << " BVH nodes" << std::endl;
        }
        else if (type != "sphere")
        {
            throw std::runtime_error("Unknown geometry type: " + type.get<std::string>());
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID.at(p["MATERIAL"].get<std::string>());
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
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.aperture = cameraData.value("APERTURE", 0.0f);
    camera.focusDistance = cameraData.value("FOCUS_DISTANCE",
        glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 360));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
