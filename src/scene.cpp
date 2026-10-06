#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <cfloat>

#include <algorithm>
#include <chrono>

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
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
        }
        else if (p["TYPE"] == "Metal" || p["TYPE"] == "Glossy")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = 1.0f;
            newMaterial.roughness = p.value("ROUGHNESS", 0.2f);
            newMaterial.metallic = (p["TYPE"] == "Metal") ? 1.0f : 0.0f;
        }

        newMaterial.texId = -1;
        // Procedural texture
        newMaterial.procTex = 0;
        if (p.contains("PROCEDURAL") && p["PROCEDURAL"] == "Checker") newMaterial.procTex = 1;
        newMaterial.procScale = p.value("PROC_SCALE", 4.0f);
        newMaterial.procColor2 = glm::vec3(0.1f);
        if (p.contains("RGB2")) {
            const auto& c2 = p["RGB2"];
            newMaterial.procColor2 = glm::vec3(c2[0], c2[1], c2[2]);
        }

        if (p.contains("TEXTURE"))
        {
            std::string baseDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            newMaterial.texId = loadTexture(baseDir + p["TEXTURE"].get<std::string>());
        }
        newMaterial.normalTexId = -1;
        if (p.contains("NORMAL_MAP"))
        {
            std::string baseDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            newMaterial.normalTexId = loadTexture(baseDir + p["NORMAL_MAP"].get<std::string>(), false);
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
            
            std::string baseDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            loadOBJ(baseDir + p["FILE"].get<std::string>(), newGeom);
        }

        // Area for light sampling (sphere assumes uniform scale)
        glm::vec3 s = newGeom.scale;
        if (newGeom.type == CUBE) newGeom.area = 2.0f * (s.x * s.y + s.y * s.z + s.z * s.x);
        else if (newGeom.type == SPHERE) newGeom.area = PI * s.x * s.x;
        else newGeom.area = 0.0f;
        if (newGeom.type != MESH && materials[newGeom.materialid].emittance > 0.0f)
            lights.push_back((int)geoms.size());

        geoms.push_back(newGeom);
    }
    cout << "Lights: " << lights.size() << endl;

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
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", 10.0f);

    state.toneMap = cameraData.value("TONEMAP", false);

    state.bloomStrength = cameraData.value("BLOOM", 0.0f);
    state.bloomThreshold = cameraData.value("BLOOM_THRESHOLD", 1.0f);
    state.bloomRadius = cameraData.value("BLOOM_RADIUS", 24);

    if (data.contains("Environment"))
    {
        const auto& env = data["Environment"];
        std::string baseDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
        std::string path = baseDir + env["FILE"].get<std::string>();
        int channels = 0;
        float* pixels = stbi_loadf(path.c_str(), &envWidth, &envHeight, &channels, 3);
        if (!pixels)
        {
            cerr << "Failed to load environment map " << path << endl;
            exit(-1);
        }
        envMap.assign((glm::vec3*)pixels, (glm::vec3*)pixels + envWidth * envHeight);
        stbi_image_free(pixels);
        envIntensity = env.value("INTENSITY", 1.0f);
        envRotation = env.value("ROTATION", 0.0f) * PI / 180.0f;
        cout << "Loaded environment map " << path << ": " << envWidth << "x" << envHeight << endl;

        // 1D CDF over env pixels, weight = luminance * sin(theta)
        int n = envWidth * envHeight;
        std::vector<double> acc(n + 1, 0.0);
        for (int y = 0; y < envHeight; ++y) {
            double sinT = sin(PI * (y + 0.5) / envHeight);
            for (int x = 0; x < envWidth; ++x) {
                const glm::vec3& c = envMap[y * envWidth + x];
                double lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
                acc[y * envWidth + x + 1] = acc[y * envWidth + x] + lum * sinT;
            }
        }
        envCdf.resize(n + 1);
        for (int i = 0; i <= n; ++i) envCdf[i] = (float)(acc[i] / acc[n]);
        envCdf[n] = 1.0f;

        // Debug: brightest pixel and its probability
        int maxI = 0; float maxP = 0.0f;
        for (int i = 0; i < n; ++i) {
            float p = envCdf[i + 1] - envCdf[i];
            if (p > maxP) { maxP = p; maxI = i; }
        }
        printf("env CDF built: max pixel (%d, %d) prob %.6f\n", maxI % envWidth, maxI / envWidth, maxP);
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


void Scene::loadOBJ(const std::string& path, Geom& geom)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err, path.c_str());
    if (!warn.empty()) cout << "OBJ warning: " << warn << endl;
    if (!ok)
    {
        cerr << "Failed to load OBJ " << path << ": " << err << endl;
        exit(-1);
    }

    geom.triStart = (int)triangles.size();
    geom.bboxMin = glm::vec3(FLT_MAX);
    geom.bboxMax = glm::vec3(-FLT_MAX);

    int badNormals = 0;

    for (const auto& shape : shapes)
    {
        const auto& idx = shape.mesh.indices;  
        for (size_t f = 0; f + 2 < idx.size(); f += 3)
        {
            glm::vec3 v[3], n[3];
            glm::vec2 uv[3];
            bool hasNormals = true;
            for (int k = 0; k < 3; k++)
            {
                const tinyobj::index_t& id = idx[f + k];
                glm::vec3 pos(attrib.vertices[3 * id.vertex_index + 0],
                    attrib.vertices[3 * id.vertex_index + 1],
                    attrib.vertices[3 * id.vertex_index + 2]);
                v[k] = glm::vec3(geom.transform * glm::vec4(pos, 1.0f));   // chenge to world coordinate

                uv[k] = id.texcoord_index >= 0
                    ? glm::vec2(attrib.texcoords[2 * id.texcoord_index], attrib.texcoords[2 * id.texcoord_index + 1])
                    : glm::vec2(0.0f);

                if (id.normal_index >= 0)
                {
                    glm::vec3 nrm(attrib.normals[3 * id.normal_index + 0],
                        attrib.normals[3 * id.normal_index + 1],
                        attrib.normals[3 * id.normal_index + 2]);
                    n[k] = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(nrm, 0.0f)));
                }
                else
                {
                    hasNormals = false;
                }
            }

            glm::vec3 faceN = glm::cross(v[1] - v[0], v[2] - v[0]);
            if (glm::length(faceN) < 1e-12f) continue;   
            if (!hasNormals)
            {
                n[0] = n[1] = n[2] = glm::normalize(faceN);
            }

            // Replace NaN/degenerate vertex normals with the face normal
            for (int k = 0; k < 3; k++)
            {
                if (!std::isfinite(n[k].x) || !std::isfinite(n[k].y) || !std::isfinite(n[k].z))
                {
                    n[k] = glm::normalize(faceN);
                    badNormals++;
                }
            }

            Triangle tri;
            tri.v0 = v[0]; tri.v1 = v[1]; tri.v2 = v[2];
            tri.n0 = n[0]; tri.n1 = n[1]; tri.n2 = n[2];
            tri.t0 = uv[0]; tri.t1 = uv[1]; tri.t2 = uv[2];

            // Tangent frame from UV derivatives (world space, since v[] is world space)
            glm::vec3 e1 = v[1] - v[0], e2 = v[2] - v[0];
            glm::vec2 d1 = uv[1] - uv[0], d2 = uv[2] - uv[0];
            float det = d1.x * d2.y - d2.x * d1.y;
            if (fabsf(det) > 1e-12f)
            {
                float r = 1.0f / det;
                tri.tangent = (e1 * d2.y - e2 * d1.y) * r;
                tri.bitangent = (e2 * d1.x - e1 * d2.x) * r;
            }
            else
            {
                tri.tangent = glm::vec3(1, 0, 0);
                tri.bitangent = glm::vec3(0, 1, 0);
            }

            triangles.push_back(tri);

            for (int k = 0; k < 3; k++)
            {
                geom.bboxMin = glm::min(geom.bboxMin, v[k]);
                geom.bboxMax = glm::max(geom.bboxMax, v[k]);
            }
        }
    }

    geom.triCount = (int)triangles.size() - geom.triStart;
    cout << "Loaded " << path << ": " << geom.triCount << " triangles" << endl;
    cout << "  bad vertex normals fixed: " << badNormals << endl;


    auto t0 = std::chrono::high_resolution_clock::now();
    int nodesBefore = (int)bvhNodes.size();
    geom.bvhRoot = buildBVH(geom.triStart, geom.triCount);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    cout << "BVH: " << (bvhNodes.size() - nodesBefore) << " nodes, built in " << ms << " ms" << endl;
}



static const int BVH_MAX_LEAF = 4;
static const int BVH_MAX_DEPTH = 60;   // GPU traversal stack has 64 entries

static glm::vec3 triCentroid(const Triangle& t)
{
    return (t.v0 + t.v1 + t.v2) / 3.0f;
}


#define BVH_USE_SAH 1                 // 1 = binned SAH split, 0 = midpoint split
static const int BVH_SAH_BINS = 16;

static float surfaceArea(glm::vec3 bmin, glm::vec3 bmax)
{
    glm::vec3 d = bmax - bmin;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Binned SAH: try BVH_SAH_BINS - 1 candidate planes per axis, return the cheapest.
// Split cost = A_left * N_left + A_right * N_right (caller normalizes by parent area).
static float findSAHSplit(const std::vector<Triangle>& tris, int first, int count,
    glm::vec3 cmin, glm::vec3 cmax, int& bestAxis, float& bestPos)
{
    float bestCost = FLT_MAX;
    bestAxis = -1;
    bestPos = 0.0f;
    for (int a = 0; a < 3; a++)
    {
        float extent = cmax[a] - cmin[a];
        if (extent <= 0.0f) continue;

        // Bin triangles by centroid and grow each bin's bounds
        glm::vec3 binMin[BVH_SAH_BINS], binMax[BVH_SAH_BINS];
        int binCount[BVH_SAH_BINS];
        for (int b = 0; b < BVH_SAH_BINS; b++)
        {
            binMin[b] = glm::vec3(FLT_MAX);
            binMax[b] = glm::vec3(-FLT_MAX);
            binCount[b] = 0;
        }
        float scale = BVH_SAH_BINS / extent;
        for (int i = first; i < first + count; i++)
        {
            const Triangle& t = tris[i];
            int b = std::min(BVH_SAH_BINS - 1, (int)((triCentroid(t)[a] - cmin[a]) * scale));
            binCount[b]++;
            binMin[b] = glm::min(binMin[b], glm::min(t.v0, glm::min(t.v1, t.v2)));
            binMax[b] = glm::max(binMax[b], glm::max(t.v0, glm::max(t.v1, t.v2)));
        }

        // Sweep from both ends: area and count on each side of every plane
        float leftArea[BVH_SAH_BINS - 1], rightArea[BVH_SAH_BINS - 1];
        int leftCount[BVH_SAH_BINS - 1], rightCount[BVH_SAH_BINS - 1];
        glm::vec3 lMin(FLT_MAX), lMax(-FLT_MAX), rMin(FLT_MAX), rMax(-FLT_MAX);
        int lSum = 0, rSum = 0;
        for (int p = 0; p < BVH_SAH_BINS - 1; p++)
        {
            lSum += binCount[p];
            lMin = glm::min(lMin, binMin[p]);
            lMax = glm::max(lMax, binMax[p]);
            leftCount[p] = lSum;
            leftArea[p] = lSum > 0 ? surfaceArea(lMin, lMax) : 0.0f;

            int q = BVH_SAH_BINS - 1 - p;
            rSum += binCount[q];
            rMin = glm::min(rMin, binMin[q]);
            rMax = glm::max(rMax, binMax[q]);
            rightCount[q - 1] = rSum;
            rightArea[q - 1] = rSum > 0 ? surfaceArea(rMin, rMax) : 0.0f;
        }

        for (int p = 0; p < BVH_SAH_BINS - 1; p++)
        {
            if (leftCount[p] == 0 || rightCount[p] == 0) continue;
            float cost = leftArea[p] * leftCount[p] + rightArea[p] * rightCount[p];
            if (cost < bestCost)
            {
                bestCost = cost;
                bestAxis = a;
                bestPos = cmin[a] + (p + 1) / scale;
            }
        }
    }
    return bestCost;
}




//#define BVH_USE_SAH 1                 // 1 = binned SAH split, 0 = midpoint split
//static const int BVH_SAH_BINS = 16;
//
//static float surfaceArea(glm::vec3 bmin, glm::vec3 bmax)
//{
//    glm::vec3 d = bmax - bmin;
//    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
//}
//
//// Binned SAH: try BVH_SAH_BINS - 1 candidate planes per axis, return the cheapest.
//// Split cost = A_left * N_left + A_right * N_right (caller normalizes by parent area).
//static float findSAHSplit(const std::vector<Triangle>& tris, int first, int count,
//    glm::vec3 cmin, glm::vec3 cmax, int& bestAxis, float& bestPos)
//{
//    float bestCost = FLT_MAX;
//    bestAxis = -1;
//    bestPos = 0.0f;
//    for (int a = 0; a < 3; a++)
//    {
//        float extent = cmax[a] - cmin[a];
//        if (extent <= 0.0f) continue;
//
//        // Bin triangles by centroid and grow each bin's bounds
//        glm::vec3 binMin[BVH_SAH_BINS], binMax[BVH_SAH_BINS];
//        int binCount[BVH_SAH_BINS];
//        for (int b = 0; b < BVH_SAH_BINS; b++)
//        {
//            binMin[b] = glm::vec3(FLT_MAX);
//            binMax[b] = glm::vec3(-FLT_MAX);
//            binCount[b] = 0;
//        }
//        float scale = BVH_SAH_BINS / extent;
//        for (int i = first; i < first + count; i++)
//        {
//            const Triangle& t = tris[i];
//            int b = std::min(BVH_SAH_BINS - 1, (int)((triCentroid(t)[a] - cmin[a]) * scale));
//            binCount[b]++;
//            binMin[b] = glm::min(binMin[b], glm::min(t.v0, glm::min(t.v1, t.v2)));
//            binMax[b] = glm::max(binMax[b], glm::max(t.v0, glm::max(t.v1, t.v2)));
//        }
//
//        // Sweep from both ends: area and count on each side of every plane
//        float leftArea[BVH_SAH_BINS - 1], rightArea[BVH_SAH_BINS - 1];
//        int leftCount[BVH_SAH_BINS - 1], rightCount[BVH_SAH_BINS - 1];
//        glm::vec3 lMin(FLT_MAX), lMax(-FLT_MAX), rMin(FLT_MAX), rMax(-FLT_MAX);
//        int lSum = 0, rSum = 0;
//        for (int p = 0; p < BVH_SAH_BINS - 1; p++)
//        {
//            lSum += binCount[p];
//            lMin = glm::min(lMin, binMin[p]);
//            lMax = glm::max(lMax, binMax[p]);
//            leftCount[p] = lSum;
//            leftArea[p] = lSum > 0 ? surfaceArea(lMin, lMax) : 0.0f;
//
//            int q = BVH_SAH_BINS - 1 - p;
//            rSum += binCount[q];
//            rMin = glm::min(rMin, binMin[q]);
//            rMax = glm::max(rMax, binMax[q]);
//            rightCount[q - 1] = rSum;
//            rightArea[q - 1] = rSum > 0 ? surfaceArea(rMin, rMax) : 0.0f;
//        }
//
//        for (int p = 0; p < BVH_SAH_BINS - 1; p++)
//        {
//            if (leftCount[p] == 0 || rightCount[p] == 0) continue;
//            float cost = leftArea[p] * leftCount[p] + rightArea[p] * rightCount[p];
//            if (cost < bestCost)
//            {
//                bestCost = cost;
//                bestAxis = a;
//                bestPos = cmin[a] + (p + 1) / scale;
//            }
//        }
//    }
//    return bestCost;
//}



int Scene::buildBVH(int first, int count)
{
    int root = (int)bvhNodes.size();
    bvhNodes.push_back(BVHNode());
    buildBVHNode(root, first, count, 0);
    return root;
}

void Scene::buildBVHNode(int nodeIdx, int first, int count, int depth)
{
    // Bounds of the triangles and of their centroids
    glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX), cmin(FLT_MAX), cmax(-FLT_MAX);
    for (int i = first; i < first + count; i++)
    {
        const Triangle& t = triangles[i];
        bmin = glm::min(bmin, glm::min(t.v0, glm::min(t.v1, t.v2)));
        bmax = glm::max(bmax, glm::max(t.v0, glm::max(t.v1, t.v2)));
        glm::vec3 c = triCentroid(t);
        cmin = glm::min(cmin, c);
        cmax = glm::max(cmax, c);
    }
    // bvhNodes may reallocate during recursion: always index, never hold a reference
    bvhNodes[nodeIdx].bboxMin = bmin;
    bvhNodes[nodeIdx].bboxMax = bmax;

    // Split along the longest axis of the centroid bounds
    glm::vec3 ext = cmax - cmin;
    int axis = 0;
    if (ext.y > ext.x) axis = 1;
    if (ext.z > ext[axis]) axis = 2;

    if (count <= BVH_MAX_LEAF || depth >= BVH_MAX_DEPTH || ext[axis] <= 0.0f)
    {
        bvhNodes[nodeIdx].leftOrFirst = first;
        bvhNodes[nodeIdx].triCount = count;
        return;
    }

    // Midpoint split by default
    float split = cmin[axis] + 0.5f * ext[axis];

#if BVH_USE_SAH
    int sahAxis;
    float sahPos;
    float sahCost = findSAHSplit(triangles, first, count, cmin, cmax, sahAxis, sahPos);
    if (sahAxis >= 0)
    {
        // Expected cost in units of one triangle test: a leaf tests all `count` triangles;
        // a split pays one traversal step plus each child's triangles weighted by
        // the chance of hitting that child (its area relative to the parent's)
        float splitCost = 1.0f + sahCost / surfaceArea(bmin, bmax);
        if (splitCost >= (float)count && count <= 16)
        {
            bvhNodes[nodeIdx].leftOrFirst = first;   // splitting does not pay off
            bvhNodes[nodeIdx].triCount = count;
            return;
        }
        axis = sahAxis;
        split = sahPos;
    }
#endif

//#if BVH_USE_SAH
//    int sahAxis;
//    float sahPos;
//    float sahCost = findSAHSplit(triangles, first, count, cmin, cmax, sahAxis, sahPos);
//    if (sahAxis >= 0)
//    {
//        // Expected cost in units of one triangle test: a leaf tests all `count` triangles;
//        // a split pays one traversal step plus each child's triangles weighted by
//        // the chance of hitting that child (its area relative to the parent's)
//        float splitCost = 1.0f + sahCost / surfaceArea(bmin, bmax);
//        if (splitCost >= (float)count && count <= 16)
//        {
//            bvhNodes[nodeIdx].leftOrFirst = first;   // splitting does not pay off
//            bvhNodes[nodeIdx].triCount = count;
//            return;
//        }
//        axis = sahAxis;
//        split = sahPos;
//    }
//#endif


    auto begin = triangles.begin() + first;
    auto end = begin + count;
    auto mid = std::partition(begin, end,
        [&](const Triangle& t) { return triCentroid(t)[axis] < split; });
    int leftCount = (int)(mid - begin);

    // Midpoint put everything on one side: fall back to a median split
    if (leftCount == 0 || leftCount == count)
    {
        leftCount = count / 2;
        std::nth_element(begin, begin + leftCount, end,
            [&](const Triangle& a, const Triangle& b) { return triCentroid(a)[axis] < triCentroid(b)[axis]; });
    }

    int left = (int)bvhNodes.size();
    bvhNodes.push_back(BVHNode());   // left child
    bvhNodes.push_back(BVHNode());   // right child, always at left + 1
    bvhNodes[nodeIdx].leftOrFirst = left;
    bvhNodes[nodeIdx].triCount = 0;

    buildBVHNode(left, first, leftCount, depth + 1);
    buildBVHNode(left + 1, first + leftCount, count - leftCount, depth + 1);
}

int Scene::loadTexture(const std::string& path, bool srgb)
{
    auto it = texCache.find(path);
    if (it != texCache.end()) return it->second;   // same file shared by several materials

    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data)
    {
        cerr << "Failed to load texture " << path << endl;
        exit(-1);
    }
    TextureInfo info{ (int)texPixels.size(), w, h };
    texPixels.reserve(texPixels.size() + (size_t)w * h);
    for (int i = 0; i < w * h; i++)
    {
        glm::vec3 s(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
        glm::vec3 c = s / 255.0f;
        texPixels.push_back(srgb ? glm::pow(c, glm::vec3(2.2f)) : c);   // normal maps stay linear
    }
    stbi_image_free(data);

    int id = (int)textures.size();
    textures.push_back(info);
    texCache[path] = id;
    cout << "Loaded texture " << path << ": " << w << "x" << h << endl;
    return id;
}
