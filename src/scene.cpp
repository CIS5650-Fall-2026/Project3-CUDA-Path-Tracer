#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <algorithm> // std::partition
#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

// for gltf
#include "tiny_gltf_v3.h"

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

/* new */
static const uint8_t* tg3AccessorPtr(const tg3_model& m, const tg3_accessor& acc, int& stride)
{
    const tg3_buffer_view& bv = m.buffer_views[acc.buffer_view];
    stride = tg3_accessor_byte_stride(&acc, &bv);
    return m.buffers[bv.buffer].data.data + bv.byte_offset + acc.byte_offset;
}

/* new */
static uint32_t readIndex(const uint8_t* p, int componentType)
{
    switch (componentType) {
    case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:  return *p;
    case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: return *(const uint16_t*)p;
    default:                                return *(const uint32_t*)p;
    }
}

static void loadGLTF(const std::string& path, Geom& geom, std::vector<Triangle>& tris)
{
    tg3_model model;
    tg3_error_stack errors;
    tg3_parse_options opts;
    tg3_error_stack_init(&errors);
    tg3_parse_options_init(&opts);

    if (tg3_parse_file(&model, &errors, path.c_str(), (uint32_t)path.size(), &opts) != TG3_OK) {
        for (uint32_t i = 0; i < tg3_errors_count(&errors); i++)
            std::cerr << "glTF: " << tg3_errors_get(&errors, i)->message << std::endl;
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        exit(-1);
    }

    geom.triStart = (int)tris.size();
    geom.bboxMin = glm::vec3(FLT_MAX);
    geom.bboxMax = glm::vec3(-FLT_MAX);

    for (uint32_t mi = 0; mi < model.meshes_count; mi++) {
        const tg3_mesh& mesh = model.meshes[mi];
        for (uint32_t pi = 0; pi < mesh.primitives_count; pi++) {
            const tg3_primitive& prim = mesh.primitives[pi];
            if (prim.mode != TG3_MODE_TRIANGLES) continue;

            // find POSITION / NORMAL accessor indices
            int posIdx = -1, nrmIdx = -1;
            for (uint32_t a = 0; a < prim.attributes_count; a++) {
                if (tg3_str_equals_cstr(prim.attributes[a].key, "POSITION")) posIdx = prim.attributes[a].value;
                else if (tg3_str_equals_cstr(prim.attributes[a].key, "NORMAL")) nrmIdx = prim.attributes[a].value;
            }
            if (posIdx < 0) continue;

            const tg3_accessor& posAcc = model.accessors[posIdx];
            int posStride;
            const uint8_t* posData = tg3AccessorPtr(model, posAcc, posStride);

            const uint8_t* nrmData = nullptr;
            int nrmStride = 0;
            if (nrmIdx >= 0) nrmData = tg3AccessorPtr(model, model.accessors[nrmIdx], nrmStride);

            auto pos = [&](uint32_t i) { return *(const glm::vec3*)(posData + i * posStride); };
            auto nrm = [&](uint32_t i) { return *(const glm::vec3*)(nrmData + i * nrmStride); };

            std::vector<uint32_t> idx;
            if (prim.indices >= 0) {
                const tg3_accessor& iAcc = model.accessors[prim.indices];
                int iStride;
                const uint8_t* iData = tg3AccessorPtr(model, iAcc, iStride);
                for (uint64_t k = 0; k < iAcc.count; k++)
                    idx.push_back(readIndex(iData + k * iStride, iAcc.component_type));
            }
            else {
                for (uint32_t k = 0; k < (uint32_t)posAcc.count; k++) idx.push_back(k);
            }

            for (size_t k = 0; k + 2 < idx.size(); k += 3) {
                Triangle t;
                t.v0 = pos(idx[k]); t.v1 = pos(idx[k + 1]); t.v2 = pos(idx[k + 2]);
                if (nrmData) {
                    t.n0 = nrm(idx[k]); t.n1 = nrm(idx[k + 1]); t.n2 = nrm(idx[k + 2]);
                }
                else {
                    glm::vec3 fn = glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
                    t.n0 = t.n1 = t.n2 = fn;
                }
                for (const glm::vec3& v : { t.v0, t.v1, t.v2 }) {
                    geom.bboxMin = glm::min(geom.bboxMin, v);
                    geom.bboxMax = glm::max(geom.bboxMax, v);
                }
                tris.push_back(t);
            }
        }
    }
    geom.triCount = (int)tris.size() - geom.triStart;
    std::cout << "Loaded " << geom.triCount << " triangles from " << path << std::endl;

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
}

/* can access tris from class directly
return index where placed */
int Scene::buildBVH(int start, int count) {
    // calc min and max bounding from start and count
    BVHNode new_node{};
    glm::vec3 cMin(FLT_MAX);
    glm::vec3 cMax(-FLT_MAX);
    for (int i = start; i < start + count; ++i) {
        // get triangle, calculate box
        const auto& t = triangles[i];
        new_node.bottom_corner = glm::min(new_node.bottom_corner, glm::min(t.v0, glm::min(t.v1, t.v2)));
        new_node.top_corner = glm::max(new_node.top_corner, glm::max(t.v0, glm::max(t.v1, t.v2)));
        auto c = (t.v0 + t.v1 + t.v2) / 3.f;
        cMin = glm::min(cMin, c);
        cMax = glm::max(cMax, c);
    }
    // quick return for leaf condition
    if (count <= 4) {
        new_node.t_start_idx = start;
        new_node.t_count = count; // not 0
        nodes.push_back(new_node);
        return (int) nodes.size() - 1; // index where node located
    }
    // calculate split based on most divergent axis to get two resultant boxes
    auto diff = cMax - cMin;
    int axis = 0; // x = 0, y = 1, z = 2
    if (diff.y > diff.x) axis = 1; // compares x and y
    if (diff.z > diff[axis]) axis = 2; // compares winner above and z
    auto split = 0.5f * (cMin[axis] + cMax[axis]);

    // sort the triangles 
    auto first = triangles.begin() + start;
    auto last = first + count;
    auto mid = std::partition(first, last, [&](const Triangle& t) {
        auto c = (t.v0 + t.v1 + t.v2) / 3.f;
        return c[axis] < split; // true gets put to the left
        });
    int lcount = (int) (mid - first);
    // fix everything on one side
    if (lcount == 0 || lcount == count) {
        lcount = count / 2;
        std::nth_element(first, first + lcount, last, [&](const Triangle& a, const Triangle& b) {
            return (a.v0[axis] + a.v1[axis] + a.v2[axis]) < (b.v0[axis] + b.v1[axis] + b.v2[axis]);
            });
    }

    int par_idx = (int)nodes.size();
    nodes.push_back(new_node);
    int left_ch = buildBVH(start, lcount); // left will always be reserved dir after parent.
    int right_ch = buildBVH(start + lcount, count - lcount);
    // go back to recently pushed node to add children in it
    nodes[par_idx].left_child = left_ch;
    nodes[par_idx].right_child = right_ch;
    return par_idx;
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
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            if (p.contains("DIFFUSE_RGB")) {
                const auto& d = p["DIFFUSE_RGB"];
                newMaterial.color = glm::vec3(d[0], d[1], d[2]);
            }
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
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
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "gltf") {
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

        if (newGeom.type == MESH) {
            loadGLTF(p["FILE"].get<std::string>(), newGeom, triangles);
            newGeom.root = buildBVH(newGeom.triStart, newGeom.triCount); // resurcisve cpu bvh creation
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

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", 10.0f);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
