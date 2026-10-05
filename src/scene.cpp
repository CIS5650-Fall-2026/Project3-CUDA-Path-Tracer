#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include "json.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <stb_image.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cfloat>
#include <filesystem>
#include <functional>
#include <cmath>

using namespace std;
using json = nlohmann::json;


Aabb emptyBounds()
{
    Aabb bounds;
    bounds.minimum = glm::vec3(FLT_MAX);
    bounds.maximum = glm::vec3(-FLT_MAX);
    return bounds;
}

void includePoint(Aabb& bounds, const glm::vec3& point)
{
    bounds.minimum = glm::min(bounds.minimum, point);
    bounds.maximum = glm::max(bounds.maximum, point);
}

void includeBounds(Aabb& bounds, const Aabb& other)
{
    bounds.minimum = glm::min(bounds.minimum, other.minimum);
    bounds.maximum = glm::max(bounds.maximum, other.maximum);
}

glm::vec3 boundsCentroid(const Aabb& bounds)
{
    return 0.5f * (bounds.minimum + bounds.maximum);
}

struct ComparePrimitiveCentroids  // used for BVH
{
    int axis;
    bool operator()(const Primitive& a, const Primitive& b) const
    {
        return boundsCentroid(a.bounds)[axis] < boundsCentroid(b.bounds)[axis];
    }
};

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

void Scene::loadObj(const std::string& filename, const glm::mat4& objectTransform, int materialId)
{
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = false;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config))
    {
        throw runtime_error("Failed to load OBJ file: " + filename + "\n" + reader.Error());
    }

    const tinyobj::attrib_t& attributes = reader.GetAttrib();
    const vector<tinyobj::shape_t>& shapes = reader.GetShapes();
    glm::mat3 normalTransform = glm::inverseTranspose(glm::mat3(objectTransform));
    int firstTriangle = static_cast<int>(triangles.size());
    int degenerateTriangleCount = 0;

    for (const tinyobj::shape_t& shape : shapes)
    {
        size_t indexOffset = 0;
        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); face++)
        {
            int vertexCount = shape.mesh.num_face_vertices[face];

            glm::vec3 positions[3];
            glm::vec3 normals[3];
            glm::vec2 uvs[3] = {};
            bool hasNormal[3] = {};

            for (int vertex = 0; vertex < 3; vertex++)
            {
                const tinyobj::index_t& index = shape.mesh.indices[indexOffset + vertex];

                positions[vertex] = glm::vec3(objectTransform * glm::vec4(
                    attributes.vertices[3 * index.vertex_index],
                    attributes.vertices[3 * index.vertex_index + 1],
                    attributes.vertices[3 * index.vertex_index + 2],
                    1.0f));

                if (index.normal_index >= 0)
                {
                    normals[vertex] = glm::normalize(normalTransform * glm::vec3(
                        attributes.normals[3 * index.normal_index],
                        attributes.normals[3 * index.normal_index + 1],
                        attributes.normals[3 * index.normal_index + 2]));
                    hasNormal[vertex] = true;
                }

                if (index.texcoord_index >= 0)
                {
                    uvs[vertex] = glm::vec2(
                        attributes.texcoords[2 * index.texcoord_index],
                        attributes.texcoords[2 * index.texcoord_index + 1]);
                }
            }
            indexOffset += vertexCount;

            glm::vec3 faceNormal = glm::cross(
                positions[1] - positions[0],
                positions[2] - positions[0]);
            if (glm::dot(faceNormal, faceNormal) <= EPSILON * EPSILON)
            {
                degenerateTriangleCount++;
                continue;
            }
            faceNormal = glm::normalize(faceNormal);

            Triangle triangle{};
            triangle.p0 = positions[0];
            triangle.p1 = positions[1];
            triangle.p2 = positions[2];
            triangle.n0 = hasNormal[0] ? normals[0] : faceNormal;
            triangle.n1 = hasNormal[1] ? normals[1] : faceNormal;
            triangle.n2 = hasNormal[2] ? normals[2] : faceNormal;
            triangle.uv0 = uvs[0];
            triangle.uv1 = uvs[1];
            triangle.uv2 = uvs[2];
            triangle.materialId = materialId;
            triangles.push_back(triangle);
        }
    }

    cout << "Loaded " << triangles.size() - firstTriangle << " triangles from " << filename;
    if (degenerateTriangleCount > 0)
    {
        cout << " (there are " << degenerateTriangleCount << " degenerate triangles)";
    }
    cout << endl;
}

void Scene::buildBvh(int leafSize)
{
    const auto start = chrono::high_resolution_clock::now();
    primitives.clear();
    bvhNodes.clear();
    bvhLeafCount = 0;
    bvhMaxDepth = 0;
    leafSize = max(1, leafSize);

    for (int i = 0; i < static_cast<int>(geoms.size()); i++)
    {
        Primitive primitive{};
        primitive.type = geoms[i].type == SPHERE ? PRIMITIVE_SPHERE : PRIMITIVE_CUBE;
        primitive.index = i;
        primitive.materialId = geoms[i].materialid;
        primitive.bounds = emptyBounds();
        for (int x = 0; x < 2; x++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int z = 0; z < 2; z++)
                {
                    glm::vec3 corner(
                        x == 0 ? -0.5f : 0.5f,
                        y == 0 ? -0.5f : 0.5f,
                        z == 0 ? -0.5f : 0.5f);
                    includePoint(primitive.bounds, glm::vec3(geoms[i].transform * glm::vec4(corner, 1.0f)));
                }
            }
        }
        primitives.push_back(primitive);
    }

    for (int i = 0; i < static_cast<int>(triangles.size()); i++)
    {
        Primitive primitive{};
        primitive.type = PRIMITIVE_TRIANGLE;
        primitive.index = i;
        primitive.materialId = triangles[i].materialId;
        primitive.bounds = emptyBounds();
        includePoint(primitive.bounds, triangles[i].p0);
        includePoint(primitive.bounds, triangles[i].p1);
        includePoint(primitive.bounds, triangles[i].p2);
        primitive.bounds.minimum -= glm::vec3(EPSILON);
        primitive.bounds.maximum += glm::vec3(EPSILON);
        primitives.push_back(primitive);
    }

    if (!primitives.empty())
    {
        bvhNodes.reserve(primitives.size() * 2);
        function<int(int, int, int)> buildNode;
        buildNode = [&](int first, int count, int depth)
        {
            int nodeIndex = static_cast<int>(bvhNodes.size());
            bvhNodes.push_back(BvhNode{});

            Aabb bounds = emptyBounds();
            Aabb centroidBounds = emptyBounds();
            for (int i = first; i < first + count; i++)
            {
                includeBounds(bounds, primitives[i].bounds);
                includePoint(centroidBounds, boundsCentroid(primitives[i].bounds));
            }

            if (count <= leafSize || depth >= 60)
            {
                BvhNode& node = bvhNodes[nodeIndex];
                node.boundsMin = bounds.minimum;
                node.boundsMax = bounds.maximum;
                node.leftmostChild = first;
                node.rightChild = -1;
                node.primitiveCount = count;
                bvhLeafCount++;
                bvhMaxDepth = max(bvhMaxDepth, depth);
                return nodeIndex;
            }

            glm::vec3 centroidExtent = centroidBounds.maximum - centroidBounds.minimum;
            int axis = 0;
            if (centroidExtent.y > centroidExtent.x) axis = 1;
            if (centroidExtent.z > centroidExtent[axis]) axis = 2;

            int middle = first + count / 2;
            nth_element(
                primitives.begin() + first,
                primitives.begin() + middle,
                primitives.begin() + first + count,
                ComparePrimitiveCentroids{axis}
            );

            int leftChild = buildNode(first, middle - first, depth + 1);
            int rightChild = buildNode(middle, first + count - middle, depth + 1);
            BvhNode& node = bvhNodes[nodeIndex];
            node.boundsMin = bounds.minimum;
            node.boundsMax = bounds.maximum;
            node.leftmostChild = leftChild;
            node.rightChild = rightChild;
            node.primitiveCount = 0;
            return nodeIndex;
        };

        buildNode(0, static_cast<int>(primitives.size()), 0);
    }

    const auto end = chrono::high_resolution_clock::now();
    bvhBuildMs = static_cast<float>(chrono::duration<double, milli>(end - start).count());
    cout << "BVH: " << bvhNodes.size() << " nodes, " << bvhLeafCount
        << " leaves, depth " << bvhMaxDepth << ", built in " << bvhBuildMs << " ms" << endl;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    materials.clear();
    geoms.clear();
    triangles.clear();
    primitives.clear();
    bvhNodes.clear();
    environment = EnvironmentMap{};

    int bvhLeafSize = 4;
    if (data.contains("Renderer"))
    {
        const auto& rendererData = data["Renderer"];
        enableCompaction = rendererData.value("COMPACTION", true);
        sortByMaterial = rendererData.value("SORT_MATERIALS", true);
        useBvh = rendererData.value("USE_BVH", true);
        bvhLeafSize = rendererData.value("BVH_LEAF_SIZE", 4);
    }

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        const std::string materialType = p["TYPE"];
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        if (materialType == "Diffuse")
        {
            newMaterial.type = MATERIAL_DIFFUSE;
        }
        else if (materialType == "Emitting")
        {
            newMaterial.type = MATERIAL_EMITTING;
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (materialType == "Specular")
        {
            newMaterial.type = MATERIAL_SPECULAR;
            newMaterial.hasReflective = 1.0f;
            newMaterial.specular.color = newMaterial.color;
            newMaterial.specular.roughness = p.value("ROUGHNESS", 0.0f);
        }
        else if (materialType == "Dielectric" || materialType == "Refractive")
        {
            newMaterial.type = MATERIAL_REFRACTIVE;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
        }
        else
        {
            throw runtime_error("Unknown material");
        }

        MatNameToID[name] = static_cast<uint32_t>(materials.size());
        materials.emplace_back(newMaterial);
    }

    const filesystem::path sceneDirectory = filesystem::path(jsonName).parent_path();
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const std::string type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        glm::vec3 translation(trans[0], trans[1], trans[2]);
        glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
        glm::vec3 objectScale(scale[0], scale[1], scale[2]);
        glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, objectScale);

        if (type == "obj")
        {
            const string materialName = p["MATERIAL"];
            auto material = MatNameToID.find(materialName);
            if (material == MatNameToID.end())
            {
                throw runtime_error("Unknown material");
            }

            filesystem::path objPath = p["FILE"].get<string>();
            if (objPath.is_relative()) objPath = sceneDirectory / objPath;
            loadObj(objPath.lexically_normal().string(), transform, static_cast<int>(material->second));
            continue;
        }

        Geom newGeom{};
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else
        {
            throw runtime_error("Unknown object type");
        }

        const string materialName = p["MATERIAL"];
        auto material = MatNameToID.find(materialName);
        if (material == MatNameToID.end())
        {
            throw runtime_error("Unknown material");
        }
        newGeom.materialid = static_cast<int>(material->second);
        newGeom.translation = translation;
        newGeom.rotation = rotation;
        newGeom.scale = objectScale;
        newGeom.transform = transform;
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    if (data.contains("Environment"))
    {
        const auto& environmentData = data["Environment"];
        filesystem::path environmentPath = environmentData["FILE"].get<string>();
        if (environmentPath.is_relative()) environmentPath = sceneDirectory / environmentPath;

        int components = 0;
        float* pixels = stbi_loadf(environmentPath.lexically_normal().string().c_str(),
            &environment.width, &environment.height, &components, 3);
        if (pixels == nullptr)
        {
            throw runtime_error("Failed to load environment map: " + environmentPath.string());
        }
        environment.pixels.resize(environment.width * environment.height);
        for (int i = 0; i < environment.width * environment.height; i++)
        {
            environment.pixels[i] = glm::vec3(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
        }
        stbi_image_free(pixels);
        environment.intensity = environmentData.value("INTENSITY", 1.0f);
        environment.rotation = environmentData.value("ROTATION", 0.0f);
        enableEnvironment = environmentData.value("ENABLED", true);
    }

    buildBvh(bvhLeafSize);

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
    camera.up = glm::normalize(glm::vec3(up[0], up[1], up[2]));
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.apertureRadius = cameraData.value("APERTURE", 0.0f);
    camera.focusDistance = cameraData.value("FOCUS_DISTANCE", glm::length(camera.lookAt - camera.position));
    enableDepthOfField = camera.apertureRadius > 0.0f;

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
