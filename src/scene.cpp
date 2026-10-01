#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "stb_image.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cfloat>

using namespace std;
using json = nlohmann::json;

TextureData loadTextureFromFile(const std::string& filename)
{
    TextureData texture;

    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &texture.width, &texture.height, &channels, 3);

    if (data == nullptr)
    {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        texture.width = 0;
        texture.height = 0;
        return texture;
    }

    int pixelCount = texture.width * texture.height;
    texture.pixels.resize(pixelCount);

    for (int i = 0; i < pixelCount; i++)
    {
        float r = data[i * 3 + 0] / 255.0f;
        float g = data[i * 3 + 1] / 255.0f;
        float b = data[i * 3 + 2] / 255.0f;

        texture.pixels[i] = glm::vec3(r, g, b);
    }

    stbi_image_free(data);

    return texture;
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

std::vector<Triangle> loadOBJ(
    const std::string& filename,
    int materialId,
    const glm::mat4& transform)
{
    std::vector<Triangle> triangles;
    std::vector<glm::vec3> vertices;

    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open OBJ: " << filename << std::endl;
        return triangles;
    }

    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string type;
        ss >> type;

        if (type == "v")
        {
            float x;
            float y;
            float z;

            ss >> x >> y >> z;

            glm::vec4 transformed = transform * glm::vec4(x, y, z, 1.0f);
            vertices.push_back(glm::vec3(transformed));
        }
        else if (type == "f")
        {
            std::vector<int> faceIndices;
            std::string token;

            while (ss >> token)
            {
                size_t slashPos = token.find('/');
                std::string vertexIndexString = token.substr(0, slashPos);
                int vertexIndex = std::stoi(vertexIndexString) - 1;
                faceIndices.push_back(vertexIndex);
            }

            for (int i = 1; i < static_cast<int>(faceIndices.size()) - 1; i++)
            {
                Triangle tri;

                tri.v0 = vertices[faceIndices[0]];
                tri.v1 = vertices[faceIndices[i]];
                tri.v2 = vertices[faceIndices[i + 1]];

                tri.normal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                tri.materialId = materialId;

                triangles.push_back(tri);
            }
        }
    }

    return triangles;
}

static void getTriangleBounds(
    const Triangle& triangle,
    glm::vec3& minBounds,
    glm::vec3& maxBounds)
{
    minBounds = glm::min(
        triangle.v0,
        glm::min(triangle.v1, triangle.v2)
    );

    maxBounds = glm::max(
        triangle.v0,
        glm::max(triangle.v1, triangle.v2)
    );
}

static glm::vec3 getTriangleCentroid(
    const Triangle& triangle)
{
    return (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
}

static int buildBVHRecursive(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int start, int end)
{
    int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(BVHNode{});

    glm::vec3 minBounds(FLT_MAX);
    glm::vec3 maxBounds(-FLT_MAX);
    glm::vec3 centroidMin(FLT_MAX);
    glm::vec3 centroidMax(-FLT_MAX);

    for (int i = start; i < end; i++)
    {
        glm::vec3 triMin;
        glm::vec3 triMax;
        getTriangleBounds(triangles[i], triMin, triMax);
        minBounds = glm::min(minBounds, triMin);
        maxBounds = glm::max(maxBounds, triMax);

        glm::vec3 centroid = getTriangleCentroid(triangles[i]);
        centroidMin = glm::min(centroidMin, centroid);
        centroidMax = glm::max(centroidMax, centroid);
    }

    int triangleCount = end - start;

    nodes[nodeIndex].minBounds = minBounds;
    nodes[nodeIndex].maxBounds = maxBounds;

    if (triangleCount <= 4)
    {
        nodes[nodeIndex].leftChild = -1;
        nodes[nodeIndex].rightChild = -1;
        nodes[nodeIndex].triangleStart = start;
        nodes[nodeIndex].triangleCount = triangleCount;
        return nodeIndex;
    }

    glm::vec3 extent = centroidMax - centroidMin;
    int axis = 0;

    if (extent.y > extent.x && extent.y >= extent.z)
    {
        axis = 1;
    }
    else if (extent.z > extent.x && extent.z > extent.y)
    {
        axis = 2;
    }

    int mid = start + triangleCount / 2;

    std::nth_element(triangles.begin() + start, triangles.begin() + mid, triangles.begin() + end, [axis](const Triangle& a, const Triangle& b) { return getTriangleCentroid(a)[axis] < getTriangleCentroid(b)[axis]; });

    int leftChild = buildBVHRecursive(triangles, nodes, start, mid);
    int rightChild = buildBVHRecursive(triangles, nodes, mid, end);

    nodes[nodeIndex].leftChild = leftChild;
    nodes[nodeIndex].rightChild = rightChild;
    nodes[nodeIndex].triangleStart = -1;
    nodes[nodeIndex].triangleCount = 0;

    return nodeIndex;
}

static void buildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes)
{
    nodes.clear();

    if (triangles.empty())
    {
        return;
    }

    buildBVHRecursive(triangles, nodes, 0, static_cast<int>(triangles.size()));

    std::cout << "BVH built: " << nodes.size() << " nodes for " << triangles.size() << " triangles" << std::endl;
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
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        newMaterial.textureType = TEXTURE_NONE;
        newMaterial.textureColor = glm::vec3(0.0f);
        newMaterial.textureScale = 1.0f;

        newMaterial.textureWidth = 0;
        newMaterial.textureHeight = 0;
        newMaterial.textureIndex = -1;
        newMaterial.textureOffset = -1;

        newMaterial.bumpWidth = 0;
        newMaterial.bumpHeight = 0;
        newMaterial.bumpOffset = -1;
        newMaterial.bumpStrength = 0.0f;

        if (p.contains("TEXTURE"))
        {
            std::string texture = p["TEXTURE"];

            if (texture == "checker")
            {
                newMaterial.textureType = TEXTURE_CHECKER;
            }
            else if (texture == "stripes")
            {
                newMaterial.textureType = TEXTURE_STRIPES;
            }

            if (p.contains("TEXTURE_RGB"))
            {
                const auto& texColor = p["TEXTURE_RGB"];
                newMaterial.textureColor = glm::vec3(texColor[0], texColor[1], texColor[2]);
            }

            if (p.contains("TEXTURE_SCALE"))
            {
                newMaterial.textureScale = p["TEXTURE_SCALE"];
            }
        }

        if (p.contains("TEXTURE_FILE"))
        {
            std::string textureFile = p["TEXTURE_FILE"];

            TextureData texture = loadTextureFromFile(textureFile);

            if (texture.width > 0 && texture.height > 0)
            {
                newMaterial.textureType = TEXTURE_FILE;
                newMaterial.textureIndex = static_cast<int>(textures.size());
                newMaterial.textureWidth = texture.width;
                newMaterial.textureHeight = texture.height;
                newMaterial.textureOffset = static_cast<int>(texturePixels.size());

                texturePixels.insert(texturePixels.end(), texture.pixels.begin(), texture.pixels.end());
                textures.push_back(texture);
            }
        }

        if (p.contains("BUMP_FILE"))
        {
            std::string bumpFile = p["BUMP_FILE"];
            TextureData bump = loadTextureFromFile(bumpFile);

            if (bump.width > 0 && bump.height > 0)
            {
                newMaterial.bumpWidth = bump.width;
                newMaterial.bumpHeight = bump.height;
                newMaterial.bumpOffset = static_cast<int>(texturePixels.size());

                texturePixels.insert(texturePixels.end(), bump.pixels.begin(), bump.pixels.end());

                if (p.contains("BUMP_STRENGTH"))
                {
                    newMaterial.bumpStrength = p["BUMP_STRENGTH"];
                }
                else
                {
                    newMaterial.bumpStrength = 1.0f;
                }
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        
        if (type == "obj")
        {
            int materialId = MatNameToID[p["MATERIAL"]];
            std::string objFile = p["FILE"];

            glm::vec3 translation(0.0f);
            glm::vec3 rotation(0.0f);
            glm::vec3 scale(1.0f);

            if (p.contains("TRANS")) translation = glm::vec3(p["TRANS"][0], p["TRANS"][1], p["TRANS"][2]);
            if (p.contains("ROTAT")) rotation = glm::vec3(p["ROTAT"][0], p["ROTAT"][1], p["ROTAT"][2]);
            if (p.contains("SCALE")) scale = glm::vec3(p["SCALE"][0], p["SCALE"][1], p["SCALE"][2]);

            glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);

            std::vector<Triangle> objTriangles = loadOBJ(objFile, materialId, transform);

            triangles.insert(triangles.end(), objTriangles.begin(), objTriangles.end());

            std::cout << "Loaded OBJ: " << objFile << " with " << objTriangles.size() << " triangles" << std::endl;

            continue;
        }
        
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "torus")
        {
            newGeom.type = TORUS;
        }
        else if (type == "menger")
        {
            newGeom.type = MENGER;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.motion = glm::vec3(0.0f);
        if (p.contains("MOTION"))
        {
            const auto& motion = p["MOTION"];
            newGeom.motion = glm::vec3(motion[0], motion[1], motion[2]);
        }
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    if (!triangles.empty())
    {
        buildBVH(triangles, bvhNodes);
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
