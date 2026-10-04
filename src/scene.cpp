#include "scene.h"

#include "bvh.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tiny_obj_loader.h"

#include <cfloat>
#include <fstream>
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
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            if (p.contains("ROUGHNESS"))
            {
                newMaterial.specular.exponent = p["ROUGHNESS"];
            }
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
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
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        if (newGeom.type == MESH)
        {
            loadFromOBJ("scenes/objs/" + p["FILE"].get<std::string>(), newGeom);
        }
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

    // Optional thin lens: APERTURE is the lens radius, and the focal distance
    // defaults to the look-at point so that whatever the camera aims at is sharp.
    camera.lensRadius = cameraData.value("APERTURE", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DIST", glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    // camera.view must be computed BEFORE camera.right, which is derived from it.
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromOBJ(const std::string& objFileName, Geom& geom)
{
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;        // fans any quads/n-gons into triangles
    config.vertex_color = false;      // unused, and it doubles the parsed size

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objFileName, config))
    {
        cout << "Couldn't load OBJ " << objFileName << ": " << reader.Error() << endl;
        exit(-1);
    }
    if (!reader.Warning().empty())
    {
        cout << "OBJ warning for " << objFileName << ": " << reader.Warning() << endl;
    }

    const tinyobj::attrib_t& attrib = reader.GetAttrib();

    geom.triangleStart = static_cast<int>(triangles.size());
    glm::vec3 bboxMin(FLT_MAX);
    glm::vec3 bboxMax(-FLT_MAX);

    for (const tinyobj::shape_t& shape : reader.GetShapes())
    {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
        {
            size_t faceVertices = shape.mesh.num_face_vertices[f];
            if (faceVertices != 3)
            {
                // config.triangulate should have prevented this; skip rather
                // than read past the face.
                indexOffset += faceVertices;
                continue;
            }

            Triangle tri{};
            glm::vec3* verts[3] = { &tri.v0, &tri.v1, &tri.v2 };
            glm::vec3* norms[3] = { &tri.n0, &tri.n1, &tri.n2 };
            bool hasNormals = true;

            for (int v = 0; v < 3; ++v)
            {
                const tinyobj::index_t& idx = shape.mesh.indices[indexOffset + v];

                *verts[v] = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);

                if (idx.normal_index >= 0)
                {
                    *norms[v] = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]);
                }
                else
                {
                    hasNormals = false;
                }

                bboxMin = glm::min(bboxMin, *verts[v]);
                bboxMax = glm::max(bboxMax, *verts[v]);
            }

            if (!hasNormals)
            {
                // No "vn" in the file: flat shade off the face normal.
                glm::vec3 faceNormal = glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
                float lengthSquared = glm::dot(faceNormal, faceNormal);
                faceNormal = lengthSquared > 0.0f
                    ? faceNormal / sqrtf(lengthSquared)
                    : glm::vec3(0.0f, 1.0f, 0.0f);   // degenerate triangle
                tri.n0 = faceNormal;
                tri.n1 = faceNormal;
                tri.n2 = faceNormal;
            }

            triangles.push_back(tri);
            indexOffset += faceVertices;
        }
    }

    geom.triangleCount = static_cast<int>(triangles.size()) - geom.triangleStart;
    if (geom.triangleCount == 0)
    {
        cout << "OBJ " << objFileName << " contributed no triangles" << endl;
        bboxMin = glm::vec3(0.0f);
        bboxMax = glm::vec3(0.0f);
    }

    // Nudge the box outwards so a face lying exactly on it can't be rejected by
    // a grazing ray's rounding.
    const float padding = 0.0001f;
    geom.bboxMin = bboxMin - padding;
    geom.bboxMax = bboxMax + padding;

    cout << "Loaded " << geom.triangleCount << " triangles from " << objFileName << endl;

    geom.bvhRoot = -1;
    if (geom.triangleCount > 0)
    {
        geom.bvhRoot = buildBVH(triangles, geom.triangleStart, geom.triangleCount, bvhNodes);
        // Linear in the mesh size, so cheap next to parsing the file.
        if (!validateBVH(triangles, geom.triangleStart, geom.triangleCount, bvhNodes, geom.bvhRoot))
        {
            exit(-1);
        }
    }
}
