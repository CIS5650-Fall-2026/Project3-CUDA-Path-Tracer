#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <sstream>

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
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"].get<float>();

            if (!(newMaterial.indexOfRefraction > 0.0f))
            {
                std::cerr << "IOR must be positive: " << name << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (p.contains("TEXTURE"))
        {
            const std::string texture = p["TEXTURE"];
            if (texture == "checker")
            {
                newMaterial.texture = TEXTURE_CHECKER;
            }
            else if (texture == "marble")
            {
                newMaterial.texture = TEXTURE_MARBLE;
            }
            else
            {
                std::cerr << "Unknown TEXTURE " << texture << " in " << name << std::endl;
                exit(EXIT_FAILURE);
            }

            const auto& col2 = p["RGB2"];
            newMaterial.color2 = glm::vec3(col2[0], col2[1], col2[2]);
            newMaterial.textureScale = p.value("TEXTURE_SCALE", 1.0f);
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
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            loadOBJ(p["FILE"], newGeom);
        }
        else if (type == "mandelbulb")
        {
            newGeom.type = MANDELBULB;
        }
        else if (type == "menger")
        {
            newGeom.type = MENGER;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);

        // Distance moved while the shutter is open.
        newGeom.velocity = glm::vec3(0.0f);
        if (p.contains("VELOCITY"))
        {
            const auto& vel = p["VELOCITY"];
            newGeom.velocity = glm::vec3(vel[0], vel[1], vel[2]);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    if (data.contains("Fog"))
    {
        const auto& fogData = data["Fog"];
        fog.density = fogData.value("DENSITY", 0.0f);
        if (fogData.contains("ALBEDO"))
        {
            const auto& albedo = fogData["ALBEDO"];
            fog.albedo = glm::vec3(albedo[0], albedo[1], albedo[2]);
        }

        if (!std::isfinite(fog.density) || fog.density < 0.0f)
        {
            std::cerr << "Fog DENSITY must be nonnegative and finite." << std::endl;
            exit(EXIT_FAILURE);
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
    
    camera.apertureRadius =
        cameraData.value("APERTURE_RADIUS", 0.0f);

    camera.focalDistance = cameraData.value(
        "FOCAL_DISTANCE",
        glm::length(camera.lookAt - camera.position));

    if (!std::isfinite(camera.apertureRadius) ||
        !std::isfinite(camera.focalDistance) ||
        camera.apertureRadius < 0.0f ||
        camera.focalDistance <= 0.0f)
    {
        std::cerr
            << "APERTURE_RADIUS must be nonnegative and "
            << "FOCAL_DISTANCE must be positive; both must be finite."
            << std::endl;
        exit(EXIT_FAILURE);
    }

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

// Simple OBJ loader, only reads v and f lines
void Scene::loadOBJ(const std::string& objName, Geom& geom)
{
    std::ifstream in(objName);
    if (!in)
    {
        std::cerr << "Couldn't open mesh " << objName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<glm::vec3> positions;
    geom.triangleStart = static_cast<int>(triangles.size());
    geom.boundsMin = glm::vec3(FLT_MAX);
    geom.boundsMax = glm::vec3(-FLT_MAX);

    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;

        if (tag == "v")
        {
            glm::vec3 v;
            ss >> v.x >> v.y >> v.z;
            positions.push_back(v);
            geom.boundsMin = glm::min(geom.boundsMin, v);
            geom.boundsMax = glm::max(geom.boundsMax, v);
        }
        else if (tag == "f")
        {
            std::vector<int> face;
            std::string vertex;
            while (ss >> vertex)
            {
                int index = std::stoi(vertex.substr(0, vertex.find('/')));
                if (index < 0)
                {
                    index += static_cast<int>(positions.size());
                }
                else
                {
                    index -= 1;
                }
                face.push_back(index);
            }

            for (size_t i = 1; i + 1 < face.size(); ++i)
            {
                triangles.push_back({
                    positions[face[0]], positions[face[i]], positions[face[i + 1]] });
            }
        }
    }

    geom.triangleCount = static_cast<int>(triangles.size()) - geom.triangleStart;

    // Traversal stack is 64, so cap depth at 32
    int maxDepth = 32;
    const char* maxDepthEnv = std::getenv("PROJECT3_BVH_MAX_DEPTH");
    if (maxDepthEnv != nullptr)
    {
        maxDepth = glm::clamp(std::atoi(maxDepthEnv), 0, 32);
    }

    const size_t nodesBefore = bvhNodes.size();
    buildBVH(geom, maxDepth);

    std::cout << "Loaded " << objName << ": " << geom.triangleCount << " triangles, "
        << bvhNodes.size() - nodesBefore << " BVH nodes" << std::endl;
}

static glm::vec3 triangleCentroid(const Triangle& tri)
{
    return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
}

// Median split BVH, triangles are reordered so leaves are contiguous
void Scene::buildBVH(Geom& geom, int maxDepth)
{
    geom.bvhRoot = static_cast<int>(bvhNodes.size());
    bvhNodes.push_back(BVHNode{});
    buildBVHNode(geom.bvhRoot, geom.triangleStart, geom.triangleCount, 0, maxDepth);
}

void Scene::buildBVHNode(int nodeIndex, int first, int count, int depth, int maxDepth)
{
    glm::vec3 boundsMin(FLT_MAX);
    glm::vec3 boundsMax(-FLT_MAX);
    glm::vec3 centroidMin(FLT_MAX);
    glm::vec3 centroidMax(-FLT_MAX);

    for (int i = first; i < first + count; ++i)
    {
        const Triangle& tri = triangles[i];
        boundsMin = glm::min(boundsMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
        boundsMax = glm::max(boundsMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
        centroidMin = glm::min(centroidMin, triangleCentroid(tri));
        centroidMax = glm::max(centroidMax, triangleCentroid(tri));
    }

    bvhNodes[nodeIndex].boundsMin = boundsMin;
    bvhNodes[nodeIndex].boundsMax = boundsMax;

    const glm::vec3 extent = centroidMax - centroidMin;
    int axis = 0;
    if (extent.y > extent[axis]) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    if (count <= 4 || depth >= maxDepth || extent[axis] <= 0.0f)
    {
        bvhNodes[nodeIndex].first = first;
        bvhNodes[nodeIndex].count = count;
        return;
    }

    const int mid = first + count / 2;
    std::nth_element(
        triangles.begin() + first,
        triangles.begin() + mid,
        triangles.begin() + first + count,
        [axis](const Triangle& a, const Triangle& b)
        {
            return triangleCentroid(a)[axis] < triangleCentroid(b)[axis];
        });

    // Children are stored next to each other
    const int left = static_cast<int>(bvhNodes.size());
    bvhNodes.push_back(BVHNode{});
    bvhNodes.push_back(BVHNode{});
    bvhNodes[nodeIndex].first = left;
    bvhNodes[nodeIndex].count = 0;

    buildBVHNode(left, first, mid - first, depth + 1, maxDepth);
    buildBVHNode(left + 1, mid, first + count - mid, depth + 1, maxDepth);
}
