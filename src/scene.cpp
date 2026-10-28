#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include <stb_image.h>

#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
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

// ---------------------------------------------------------------------------
// glTF loading (v1: one mesh, one primitive, POSITION / NORMAL / TEXCOORD_0 / indices)
//
// attribute -> accessor -> bufferView -> buffer (.bin); reading is just
// start byte + i * stride, reinterpreted as float / uint16 / uint32
// Ignores node transforms and any primitive after the first
// ---------------------------------------------------------------------------

// Size in bytes of one component (5126 = float, 5123 = uint16, 5125 = uint32, 5121 = uint8)
static int gltfComponentSize(int componentType)
{
    switch (componentType)
    {
    case 5126: return 4;
    case 5125: return 4;
    case 5123: return 2;
    case 5121: return 1;
    default:   return 0;
    }
}

// Number of components per element ("SCALAR" = 1, "VEC2" = 2, "VEC3" = 3, ...)
static int gltfTypeCount(const std::string& type)
{
    if (type == "SCALAR") return 1;
    if (type == "VEC2")   return 2;
    if (type == "VEC3")   return 3;
    if (type == "VEC4")   return 4;
    return 0;
}

// Returns a pointer to the first byte of accessor `accessorIndex` inside `bytes`,
// and fills in the per-element stride, so callers can walk elements with
//     base + i * stride
static const unsigned char* gltfAccessorBase(
    const json& gltf,
    const std::vector<unsigned char>& bytes,
    int accessorIndex,
    int& outCount,
    int& outStride,
    int& outComponentType)
{
    const json& accessor = gltf["accessors"][accessorIndex];
    const json& view = gltf["bufferViews"][accessor["bufferView"].get<int>()];

    outCount = accessor["count"];
    outComponentType = accessor["componentType"];
    int elementSize = gltfComponentSize(outComponentType) * gltfTypeCount(accessor["type"]);

    // a bufferView may have "byteStride"; if absent the data is tightly packed and the stride is elementSize
    outStride = view.value("byteStride", elementSize);

    // the start byte is the sum of two optional offsets:
    //   bufferView.byteOffset  (default 0)
    //   accessor.byteOffset    (default 0)
    size_t start = view.value("byteOffset", 0) + accessor.value("byteOffset", 0);

    return bytes.data() + start;
}

// Reads a float VECn accessor into a flat vector [x0,y0,z0, x1,y1,z1, ...]
static std::vector<float> gltfReadFloats(
    const json& gltf,
    const std::vector<unsigned char>& bytes,
    int accessorIndex)
{
    int count, stride, componentType;
    const unsigned char* base = gltfAccessorBase(gltf, bytes, accessorIndex, count, stride, componentType);
    int n = gltfTypeCount(gltf["accessors"][accessorIndex]["type"]);

    std::vector<float> out;
    out.reserve(count * n);
    for (int i = 0; i < count; i++)
    {
        const float* p = reinterpret_cast<const float*>(base + i * stride);
        for (int c = 0; c < n; c++) out.push_back(p[c]);
    }
    return out;
}

// Reads a SCALAR index accessor (uint16 or uint32) into ints
static std::vector<int> gltfReadIndices(
    const json& gltf,
    const std::vector<unsigned char>& bytes,
    int accessorIndex)
{
    int count, stride, componentType;
    const unsigned char* base = gltfAccessorBase(gltf, bytes, accessorIndex, count, stride, componentType);

    std::vector<int> out;
    out.reserve(count);
    for (int i = 0; i < count; i++)
    {
        const unsigned char* p = base + i * stride;
        int value;

        if (componentType == 5123) {
            value = *reinterpret_cast<const uint16_t*>(p);
        }
        else { // 5125 
            value = *reinterpret_cast<const uint32_t*>(p);
        }

        out.push_back(value);
    }
    return out;
}

// Deepest split the traversal stack can follow
// An interior node at depth k leaves k pending siblings on the stack and pushes two more,
// which needs k + 2 <= BVH_STACK_SIZE (32 in pathtrace.cu), and interior nodes sit above bvhMaxDepth
static const int bvhDepthCeiling = 31;

static glm::vec3 triangleCentroid(const Triangle& tri)
{
    return (tri.vertices[0] + tri.vertices[1] + tri.vertices[2]) / 3.0f;
}

// Builds the hierarchy over triangles[triStart, triStart + triCount) and returns the
// index of its root in bvhNodes. Triangles are reordered in place, so that every node
// owns one contiguous slice
int Scene::buildBVH(int triStart, int triCount, int depth)
{
    // reserve the slot first so a parent always sits before its children; the node is
    // filled through a local copy because bvhNodes can reallocate during the recursion
    int nodeIndex = (int)bvhNodes.size();
    bvhNodes.push_back(BVHNode());

    BVHNode node;
    node.aabbMin = glm::vec3(FLT_MAX);
    node.aabbMax = glm::vec3(-FLT_MAX);
    node.left = -1;
    node.right = -1;
    node.triStart = triStart;
    node.triCount = triCount;

    // bounds of the centroids, used to choose where to split
    glm::vec3 centroidMin(FLT_MAX);
    glm::vec3 centroidMax(-FLT_MAX);

    // the box holds every vertex; the centroid bounds only hold one point per triangle
    for (int i = triStart; i < triStart + triCount; i++)
    {
        const Triangle& tri = triangles[i];
        for (int k = 0; k < 3; k++)
        {
            node.aabbMin = glm::min(node.aabbMin, tri.vertices[k]);
            node.aabbMax = glm::max(node.aabbMax, tri.vertices[k]);
        }
        centroidMin = glm::min(centroidMin, triangleCentroid(tri));
        centroidMax = glm::max(centroidMax, triangleCentroid(tri));
    }

    if (triCount > bvhLeafSize && depth < bvhMaxDepth)
    {
        // split along the axis on which the centroids are spread the widest
        glm::vec3 extent = centroidMax - centroidMin;
        int axis = 0;
        if (extent.y > extent[axis]) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        // median split: the half with the smaller centroids moves to the front
        // nth_element is enough, the order inside each half does not matter
        int mid = triCount / 2;
        std::nth_element(
            triangles.begin() + triStart,
            triangles.begin() + triStart + mid,
            triangles.begin() + triStart + triCount,
            [axis](const Triangle& a, const Triangle& b)
            {
                return triangleCentroid(a)[axis] < triangleCentroid(b)[axis];
            });

        // mid counts from the start of this slice, not from the start of the array
        // triCount 0 marks the node as interior
        node.left = buildBVH(triStart, mid, depth + 1);
        node.right = buildBVH(triStart + mid, triCount - mid, depth + 1);
        node.triCount = 0;
    }

    bvhNodes[nodeIndex] = node;
    return nodeIndex;
}

void Scene::loadGLTF(const std::string& path, Geom& geom)
{
    cout << "Loading glTF " << path << " ..." << endl;

    // 1. Parse the .gltf JSON 
    std::ifstream f(path);
    if (!f) { cout << "  could not open " << path << endl; exit(-1); }
    json gltf = json::parse(f);

    // 2. Read the .bin buffer into memory; its uri is relative to the .gltf directory
    std::string dir = path.substr(0, path.find_last_of("/\\") + 1);
    std::string binPath = dir + gltf["buffers"][0]["uri"].get<std::string>();
    std::ifstream binFile(binPath, std::ios::binary);
    if (!binFile) { cout << "  could not open " << binPath << endl; exit(-1); }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(binFile)), {});
    cout << "  .bin size " << bytes.size() << " bytes (gltf says "
         << gltf["buffers"][0]["byteLength"] << ")" << endl;

    // 3. Every primitive of every mesh goes into this one geom, because a model is often
    // exported in parts (pot, soil, plant) that share one material and one coordinate space
    // Node transforms are not applied, so the parts must already sit where they belong
    geom.triStart = (int)triangles.size();
    glm::vec3 aabbMin(FLT_MAX);
    glm::vec3 aabbMax(-FLT_MAX);

    for (const json& mesh : gltf["meshes"])
    {
        for (const json& prim : mesh["primitives"])
        {
            // values are accessor indices
            int posAcc = prim["attributes"]["POSITION"];
            int nrmAcc = prim["attributes"]["NORMAL"];
            int uvAcc  = prim["attributes"].value("TEXCOORD_0", -1);   // -1 if the mesh has no UVs
            int idxAcc = prim["indices"];

            std::vector<float> positions = gltfReadFloats(gltf, bytes, posAcc);
            std::vector<float> normals   = gltfReadFloats(gltf, bytes, nrmAcc);
            std::vector<float> uvs       = (uvAcc >= 0) ? gltfReadFloats(gltf, bytes, uvAcc) : std::vector<float>();
            std::vector<int>   indices   = gltfReadIndices(gltf, bytes, idxAcc);

            cout << "  " << positions.size() / 3 << " vertices, " << indices.size() / 3 << " triangles" << endl;

            // 4. Assemble triangles in WORLD space so intersection and AABB/BVH need no inverse transform
            for (size_t t = 0; t + 2 < indices.size(); t += 3)
            {
                Triangle tri;
                for (int k = 0; k < 3; k++)
                {
                    int vi = indices[t + k];

                    // pull vertex vi out of the flat arrays
                    glm::vec3 p(positions[vi * 3], positions[vi * 3 + 1], positions[vi * 3 + 2]);
                    glm::vec3 n(normals[vi * 3], normals[vi * 3 + 1], normals[vi * 3 + 2]);
                    glm::vec2 uv(0.0f);
                    if (!uvs.empty())  uv = glm::vec2(uvs[vi * 2], uvs[vi * 2 + 1]);

                    // w = 1 for a point, w = 0 for a direction; normals use the inverse transpose
                    p = glm::vec3(geom.transform * glm::vec4(p, 1.0f));
                    n = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(n, 0.0f)));

                    tri.vertices[k] = p;
                    tri.normals[k] = n;
                    tri.uvs[k] = uv;
                    aabbMin = glm::min(aabbMin, p);
                    aabbMax = glm::max(aabbMax, p);
                }
                triangles.push_back(tri);
            }
        }
    }
    geom.triCount = (int)triangles.size() - geom.triStart;
    geom.aabbMin = aabbMin;
    geom.aabbMax = aabbMax;

    int nodesBefore = (int)bvhNodes.size();
    geom.bvhRoot = buildBVH(geom.triStart, geom.triCount, 0);
    cout << "  " << bvhNodes.size() - nodesBefore << " BVH nodes" << endl;
}

// Loads one image, appends its pixels to `texels` and returns its index in `textures`
int Scene::loadTexture(const std::string& path)
{
    cout << "Loading texture " << path << " ..." << endl;

    // the last argument forces 3 channels, whatever the file has
    int w, h, channels;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) { cout << "  could not load " << path << endl; exit(-1); }

    TextureInfo info;
    // this image starts where the texel array currently ends
    info.offset = texels.size();
    info.width = w;
    info.height = h;

    // stb gives 3 bytes per pixel in 0..255, row by row from the top; store floats in 0..1
    for (int i = 0; i < w * h; i++) {
        glm::vec3 rgb(data[3 * i] / 255.0, data[3 * i + 1] / 255.0, data[3 * i + 2] / 255.0);
        texels.push_back(rgb);
    }

    stbi_image_free(data);
    textures.push_back(info);

    cout << "  " << w << " x " << h << ", " << channels << " channels in file, first texel "
         << glm::to_string(texels[info.offset]) << endl;
    return (int)textures.size() - 1;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    // paths inside the scene file are relative to the scene file
    std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);

    // optional BVH build settings, read before any mesh is loaded because the tree is built then
    if (data.contains("BVH"))
    {
        bvhLeafSize = data["BVH"].value("LEAF_SIZE", bvhLeafSize);
        bvhMaxDepth = data["BVH"].value("MAX_DEPTH", bvhMaxDepth);
    }
    bvhLeafSize = glm::max(bvhLeafSize, 1);
    bvhMaxDepth = glm::clamp(bvhMaxDepth, 0, bvhDepthCeiling);
    cout << "BVH leaf size " << bvhLeafSize << ", depth limit " << bvhMaxDepth << endl;

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        newMaterial.surface.image = -1;

        const std::string type = p["TYPE"];
        if (type == "Diffuse")
        {
            newMaterial.type = DIFFUSE;
        }
        else if (type == "Emitting")
        {
            newMaterial.type = EMISSIVE;
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (type == "Specular")
        {
            newMaterial.type = SPECULAR;
            // Phong exponent of the lobe; leaving it out keeps a perfect mirror
            newMaterial.exponent = p.value("EXPONENT", 0.0f);
        }
        else if (type == "Refractive")
        {
            newMaterial.type = DIELECTRIC;
            newMaterial.ior = p["IOR"];
        }
        else
        {
            cout << "unknown material type " << type << " on " << name << endl;
            exit(-1);
        }

        // optional computed pattern for the base color; RGB is the tile color
        if (p.contains("PROCEDURAL") && p["PROCEDURAL"] == "tiles")
        {
            newMaterial.surface.procedural = PROC_TILES;
            // one number applies to both axes; [u, v] sets them apart, which a surface
            // that is not square needs to get square tiles and grout of one thickness
            newMaterial.tiles.count = glm::vec2(8.0f);
            if (p.contains("TILES"))
            {
                const auto& t = p["TILES"];
                newMaterial.tiles.count = t.is_array() ? glm::vec2(t[0], t[1]) : glm::vec2(t.get<float>());
            }
            newMaterial.tiles.grout = glm::vec2(0.04f);
            if (p.contains("GROUT"))
            {
                const auto& g = p["GROUT"];
                newMaterial.tiles.grout = g.is_array() ? glm::vec2(g[0], g[1]) : glm::vec2(g.get<float>());
            }
            newMaterial.tiles.groutColor = glm::vec3(0.25f);
            if (p.contains("GROUT_RGB"))
            {
                const auto& g = p["GROUT_RGB"];
                newMaterial.tiles.groutColor = glm::vec3(g[0], g[1], g[2]);
            }
        }
        // optional bump strength; the height comes from the material's procedural pattern
        newMaterial.surface.bump = p.value("BUMP", 0.0f);
        // optional image for the base color, any material type
        if (p.contains("TEXTURE"))
        {
            newMaterial.surface.image = loadTexture(sceneDir + p["TEXTURE"].get<std::string>());
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
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else
        {
            newGeom.type = MESH;
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
            loadGLTF(sceneDir + p["FILE"].get<std::string>(), newGeom);
        }

        // direct lighting samples points on emissive boxes, so remember where they are
        if (newGeom.type == CUBE && materials[newGeom.materialid].type == EMISSIVE)
        {
            lights.push_back((int)geoms.size());
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
    // thin lens; both are optional and a scene without them renders as before
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", glm::length(camera.lookAt - camera.position));

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
