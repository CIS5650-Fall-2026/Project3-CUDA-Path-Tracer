#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

// Simple OBJ reader. Only vertex positions ("v") and faces ("f") are used;
// normals, UVs and materials ignored
static bool loadOBJ(const string& path, vector<glm::vec3>& positions, vector<glm::ivec3>& faces)
{
    ifstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    string line;
    while (getline(file, line))
    {
        istringstream stream(line);
        string tag;
        stream >> tag;

        if (tag == "v")
        {
            glm::vec3 p;
            stream >> p.x >> p.y >> p.z;
            positions.push_back(p);
        }
        else if (tag == "f")
        {
            // Keep only the vertex index of each corner
            vector<int> corners;
            string corner;
            while (stream >> corner)
            {
                int index = stoi(corner.substr(0, corner.find('/')));

                // OBJ counts from 1, and negative means "count back from the last vertex"
                if (index > 0)
                {
                    corners.push_back(index - 1);
                }
                else
                {
                    corners.push_back((int)positions.size() + index);
                }
            }

            // Split quads and bigger polygons into a fan of triangles
            for (size_t i = 1; i + 1 < corners.size(); ++i)
            {
                faces.push_back(glm::ivec3(corners[0], corners[i], corners[i + 1]));
            }
        }
    }
    return true;
}

// avoid NaN on a zero vector
static glm::vec3 safeNormalize(glm::vec3 v)
{
    float len = glm::length(v);
    if (len < 1e-12f)
    {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }
    return v / len;
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
            newMaterial.hasReflective = 1.0f;

            // Rough metal: ROUGHNESS 0 = perfect mirror. Turn roughness into a Phong
            // exponent (common mapping n = 2 / r^2 - 2): rougher = blurrier reflection
            if (p.contains("ROUGHNESS"))
            {
                float roughness = p["ROUGHNESS"];
                if (roughness > 0.0f)
                {
                    newMaterial.specular.exponent = glm::max(2.0f / (roughness * roughness) - 2.0f, 0.01f);
                }
            }
        }
        else if (p["TYPE"] == "Refractive")
        {
            // Glass-like: RGB tints the light going through, IOR defaults to glass
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = 1.5f;
            if (p.contains("IOR"))
            {
                newMaterial.indexOfRefraction = p["IOR"];
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // Mesh files are looked up relative to the scene file's folder
    const string sceneFolder = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);

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

        // Motion blur: optional movement during the shutter, no VELOCITY = standing still
        newGeom.velocity = glm::vec3(0.0f);
        if (p.contains("VELOCITY"))
        {
            const auto& vel = p["VELOCITY"];
            newGeom.velocity = glm::vec3(vel[0], vel[1], vel[2]);
        }

        if (newGeom.type == MESH)
        {
            // Smooth shading by default; set "SMOOTH": false for a faceted look
            bool smoothNormals = true;
            if (p.contains("SMOOTH"))
            {
                smoothNormals = p["SMOOTH"];
            }
            loadMesh(sceneFolder + p["FILE"].get<string>(), smoothNormals, newGeom);
        }

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

    // Depth of field is optional. No lens radius means a pinhole camera,
    // and by default we focus on the look-at point.
    camera.lensRadius = 0.0f;
    if (cameraData.contains("LENS_RADIUS"))
    {
        camera.lensRadius = cameraData["LENS_RADIUS"];
    }
    camera.focalDistance = glm::length(camera.lookAt - camera.position);
    if (cameraData.contains("FOCAL_DIST"))
    {
        camera.focalDistance = cameraData["FOCAL_DIST"];
    }

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

void Scene::loadMesh(const std::string& objPath, bool smoothNormals, Geom& mesh)
{
    vector<glm::vec3> positions;
    vector<glm::ivec3> faces;
    if (!loadOBJ(objPath, positions, faces) || faces.empty())
    {
        cout << "Couldn't load mesh " << objPath << endl;
        exit(-1);
    }

    // Center the model and scale its longest side to 1 so TRANS/ROTAT/SCALE mean the same thing
    glm::vec3 lo(FLT_MAX);
    glm::vec3 hi(-FLT_MAX);
    for (const glm::vec3& v : positions)
    {
        lo = glm::min(lo, v);
        hi = glm::max(hi, v);
    }
    glm::vec3 center = 0.5f * (lo + hi);
    glm::vec3 size = hi - lo;
    float longestSide = glm::max(size.x, glm::max(size.y, size.z));

    // Move everything into world space once here, so the GPU never has to transform rays for meshes
    for (glm::vec3& v : positions)
    {
        glm::vec3 unitSpace = (v - center) / longestSide;
        v = glm::vec3(mesh.transform * glm::vec4(unitSpace, 1.0f));
    }

    // Vertex normal = sum of the face normals around it
    vector<glm::vec3> vertexNormals(positions.size(), glm::vec3(0.0f));
    for (const glm::ivec3& f : faces)
    {
        glm::vec3 faceNormal = glm::cross(positions[f.y] - positions[f.x], positions[f.z] - positions[f.x]);
        vertexNormals[f.x] += faceNormal;
        vertexNormals[f.y] += faceNormal;
        vertexNormals[f.z] += faceNormal;
    }

    mesh.triStart = triangles.size();
    mesh.triCount = faces.size();
    mesh.bboxMin = glm::vec3(FLT_MAX);
    mesh.bboxMax = glm::vec3(-FLT_MAX);

    for (const glm::ivec3& f : faces)
    {
        Triangle tri;
        tri.v0 = positions[f.x];
        tri.v1 = positions[f.y];
        tri.v2 = positions[f.z];

        if (smoothNormals)
        {
            tri.n0 = safeNormalize(vertexNormals[f.x]);
            tri.n1 = safeNormalize(vertexNormals[f.y]);
            tri.n2 = safeNormalize(vertexNormals[f.z]);
        }
        else
        {
            glm::vec3 flat = safeNormalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
            tri.n0 = flat;
            tri.n1 = flat;
            tri.n2 = flat;
        }
        triangles.push_back(tri);

        // Update the bounding box used for culling
        mesh.bboxMin = glm::min(mesh.bboxMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
        mesh.bboxMax = glm::max(mesh.bboxMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
    }

    cout << "Loaded " << objPath << ": " << faces.size() << " triangles" << endl;
}
