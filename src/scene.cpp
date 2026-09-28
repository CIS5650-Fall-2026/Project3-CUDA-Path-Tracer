#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
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

void Scene::loadMesh(const std::string& objPath, const glm::mat4& transform, int materialId)
{
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objPath, tinyobj::ObjReaderConfig()))
    {
        cout << "Failed to load mesh " << objPath << ": " << reader.Error() << endl;
        exit(-1);
    }

    const glm::mat3 normalMatrix = glm::mat3(glm::inverseTranspose(transform));
    const tinyobj::attrib_t& attrib = reader.GetAttrib();

    auto position = [&](int index)
    {
        glm::vec4 p(attrib.vertices[3 * index + 0],
                    attrib.vertices[3 * index + 1],
                    attrib.vertices[3 * index + 2], 1.0f);
        return glm::vec3(transform * p);
    };

    size_t triangleCount = 0;
    for (const auto& shape : reader.GetShapes())
    {
        // ObjReader triangulates for us
        for (size_t f = 0; f + 2 < shape.mesh.indices.size(); f += 3)
        {
            const tinyobj::index_t& i0 = shape.mesh.indices[f + 0];
            const tinyobj::index_t& i1 = shape.mesh.indices[f + 1];
            const tinyobj::index_t& i2 = shape.mesh.indices[f + 2];

            Triangle tri{};
            tri.v0 = position(i0.vertex_index);
            tri.v1 = position(i1.vertex_index);
            tri.v2 = position(i2.vertex_index);
            tri.materialId = materialId;

            if (i0.normal_index >= 0 && !attrib.normals.empty())
            {
                auto objNormal = [&](int index)
                {
                    return glm::normalize(normalMatrix * glm::vec3(
                        attrib.normals[3 * index + 0],
                        attrib.normals[3 * index + 1],
                        attrib.normals[3 * index + 2]));
                };
                tri.n0 = objNormal(i0.normal_index);
                tri.n1 = objNormal(i1.normal_index);
                tri.n2 = objNormal(i2.normal_index);
            }
            else
            {
                glm::vec3 faceNormal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                tri.n0 = tri.n1 = tri.n2 = faceNormal;
            }

            triangles.push_back(tri);
            triangleCount++;
        }
    }

    cout << "Loaded " << triangleCount << " triangles from " << objPath << endl;
}

namespace
{
    struct TriangleBounds
    {
        glm::vec3 bboxMin;
        glm::vec3 bboxMax;
        glm::vec3 centroid;
    };

    void growBounds(glm::vec3& lo, glm::vec3& hi, const glm::vec3& p)
    {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
}

// median split on the widest centroid axis
void Scene::buildBVH()
{
    bvhNodes.clear();
    if (triangles.empty())
    {
        return;
    }

    std::vector<TriangleBounds> bounds(triangles.size());
    for (size_t i = 0; i < triangles.size(); i++)
    {
        const Triangle& t = triangles[i];
        glm::vec3 lo = glm::min(t.v0, glm::min(t.v1, t.v2));
        glm::vec3 hi = glm::max(t.v0, glm::max(t.v1, t.v2));
        // pad so flat boxes still pass the slab test
        bounds[i] = { lo - 1e-5f, hi + 1e-5f, (lo + hi) * 0.5f };
    }

    std::vector<int> order(triangles.size());
    std::iota(order.begin(), order.end(), 0);

    const int maxLeafSize = 4;
    bvhNodes.reserve(triangles.size() * 2);

    // a range of order[] plus the node it fills
    struct BuildTask
    {
        int start;
        int end;
        int nodeIndex;
    };
    bvhNodes.push_back(BVHNode{});
    std::vector<BuildTask> tasks{ { 0, (int)order.size(), 0 } };

    while (!tasks.empty())
    {
        BuildTask task = tasks.back();
        tasks.pop_back();

        glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
        glm::vec3 centroidLo(FLT_MAX), centroidHi(-FLT_MAX);
        for (int i = task.start; i < task.end; i++)
        {
            growBounds(lo, hi, bounds[order[i]].bboxMin);
            growBounds(lo, hi, bounds[order[i]].bboxMax);
            growBounds(centroidLo, centroidHi, bounds[order[i]].centroid);
        }

        BVHNode& node = bvhNodes[task.nodeIndex];
        node.bboxMin = lo;
        node.bboxMax = hi;

        const int count = task.end - task.start;
        glm::vec3 extent = centroidHi - centroidLo;
        int axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);

        if (count <= maxLeafSize || extent[axis] < 1e-8f)
        {
            node.leftChild = -1;
            node.firstTri = task.start;
            node.triCount = count;
            continue;
        }

        const int mid = task.start + count / 2;
        std::nth_element(order.begin() + task.start, order.begin() + mid, order.begin() + task.end,
            [&](int a, int b)
            {
                return bounds[a].centroid[axis] < bounds[b].centroid[axis];
            });

        const int leftIndex = (int)bvhNodes.size();
        bvhNodes.push_back(BVHNode{});
        bvhNodes.push_back(BVHNode{});
        // node is dangling after those push_backs
        bvhNodes[task.nodeIndex].leftChild = leftIndex;
        bvhNodes[task.nodeIndex].triCount = 0;

        tasks.push_back({ task.start, mid, leftIndex });
        tasks.push_back({ mid, task.end, leftIndex + 1 });
    }

    // leaves index into order[], so reorder the triangles to match
    std::vector<Triangle> reordered(triangles.size());
    for (size_t i = 0; i < order.size(); i++)
    {
        reordered[i] = triangles[order[i]];
    }
    triangles.swap(reordered);

    cout << "Built BVH: " << bvhNodes.size() << " nodes over "
         << triangles.size() << " triangles" << endl;
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
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.specular.color = newMaterial.color;
            // chance of taking the specular lobe instead of the diffuse one
            newMaterial.hasReflective = p.value("REFLECTIVE", 1.0f);
            // 0 = perfect mirror
            float roughness = p.value("ROUGHNESS", 0.0f);
            newMaterial.specular.exponent = roughness > 0.0f
                ? 2.0f / (roughness * roughness) - 2.0f
                : 0.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.55f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
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
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.type == MESH)
        {
            // mesh paths are relative to the scene file, not the cwd
            std::string meshFile = p["FILE"];
            size_t slash = jsonName.find_last_of("/\\");
            if (slash != std::string::npos)
            {
                meshFile = jsonName.substr(0, slash + 1) + meshFile;
            }

            // meshes live in the triangle/BVH arrays, not the geom list
            loadMesh(meshFile, newGeom.transform, newGeom.materialid);
            continue;
        }

        geoms.push_back(newGeom);
    }

    buildBVH();

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", 10.0f);
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
