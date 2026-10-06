#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "scene.h"
#include "stb_image.h"

#include "stb_image_write.h"  

struct DecodedImage {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> pixels;
};

// trouble with new version of gltf's image decoding 
// grab uris and decode with stb_image instead
static bool decodeImg(const tg3_model& model, int imgIdx, const std::string& baseDir, DecodedImage& out) {
    const tg3_image& img = model.images[imgIdx];
    const uint8_t* bytes = nullptr; 
    size_t len = 0;
    std::vector<uint8_t> imgLoad;

    if (img.uri.data && img.uri.len > 0) { // uri detected 
        std::string uri(img.uri.data, img.uri.len);
        // storing texture files externally 
        std::ifstream f(baseDir + "/" + uri, std::ios::binary);
        if (!f) return false; // return if file couldn't open
        imgLoad.assign(std::istreambuf_iterator<char>(f), {});
    }

    // store loaded info
    bytes = imgLoad.data(); 
    len = imgLoad.size();

    int n;
    // returns pointer to raw pixels
    stbi_uc* px = stbi_load_from_memory(bytes, (int)len, &out.w, &out.h, &n, 4);
    if (!px) return false; // return if decoding failed

    // assign info to DecodedImage struct
    out.pixels.assign(px, px + (size_t)out.w * out.h * 4);

    // free pointer 
    stbi_image_free(px);

    // successful decoding
    return true;
}

Scene::Scene(std::string filename)
{
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        std::cout << "Couldn't read from " << filename << std::endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json sceneData= json::parse(f);
    const auto& materialsData = sceneData["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["ALBEDO"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.0f;
            newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["ALBEDO"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["ALBEDO"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.type = SPECULAR;
        }
        else if (p["TYPE"] == "MetallicWorkflow") {
            const auto& col = p["ALBEDO"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.metallic = p["METALLIC"];
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.type = METALLICWORKFLOW;
        } 
        else if (p["TYPE"] == "Dielectric") {
            const auto& col = p["ALBEDO"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.refractionIndex = p["IOR"];
            const auto& transmit = p["ABSORPTION"];
            newMaterial.absorption = glm::vec3(transmit[0], transmit[1], transmit[2]);
            newMaterial.type = DIELECTRIC;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    std::vector<json> models = {};

    const auto& objectsData = sceneData["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

        if (type == "mesh") {
            models.push_back(p);
            continue;
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
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

        geoms.push_back(newGeom);
        // if geom is a light
        if (materials[newGeom.materialid].emittance > 0.0f) {
            lights.push_back(newGeom);
        }
    }

    for (const auto& model : models) {
        // fill with triangle data 
        gltfLoad(model, MatNameToID);
    }


    // load environment map 
    const auto& envMapData = sceneData["Environment Map"];
    std::string envMapFile = envMapData["FILEPATH"];
    if (envMapFile != "none") {
        const char* envMap = envMapFile.c_str();
        float* map;
        int width;
        int height;
        const char* err = nullptr;

        int ret = LoadEXR(&map, &width, &height, envMap, &err);

        if (ret != TINYEXR_SUCCESS) {
            if (err) {
                fprintf(stderr, "ERR : %s\n", err);
                FreeEXRErrorMessage(err);
            }
        }
        else {
            envMapPixels.assign(map, map + (width * height * 4));
            envMapWidth = width;
            envMapHeight = height;
            free(map);
        }
    }

    const auto& cameraData = sceneData["Camera"];
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

void Scene::gltfLoad(const json& modelData, std::unordered_map<std::string, uint32_t> MatNameToID) {
    // first load transforms 
    const auto& trans = modelData["TRANS"];
    const auto& rot = modelData["ROTAT"];
    const auto& scl = modelData["SCALE"];

    glm::vec3 translate = glm::vec3(trans[0], trans[1], trans[2]);
    glm::vec3 rotate = glm::vec3(rot[0], rot[1], rot[2]);
    glm::vec3 scale = glm::vec3(scl[0], scl[1], scl[2]);
    glm::mat4 transform = utilityCore::buildTransformationMatrix(
        translate, rotate, scale);
    glm::mat4 inverse = glm::inverse(transform);
    glm::mat4 inverseTranspose = glm::inverseTranspose(transform);

    // tiny gltf
    tg3_parse_options opts; // configuration
    tg3_error_stack errors; // stores errors/warnings encountered during parsing
    tg3_model model; // parsed mode

    // fill with default settings
    tg3_parse_options_init(&opts);

    // initialize error stack
    tg3_error_stack_init(&errors);

    // parse file
    std::string jsonpath = modelData["FILEPATH"];
    const char* filepath = jsonpath.c_str();
    uint32_t filelen = jsonpath.size();

    std::string baseDir = jsonpath.substr(0, jsonpath.find_last_of("/\\"));

    tg3_error_code err = tg3_parse_file(&model, &errors, filepath, filelen, &opts);
    if (err != TG3_OK) {
        // print errors
        for (uint32_t i = 0; i < errors.count; i++) {
            fprintf(stderr, "[%d] %s\n", (int)errors.entries[i].severity,
                errors.entries[i].message ? errors.entries[i].message : "(null)");
        }
    }

    std::unordered_map<int, int> idxToTex;

    for (int i = 0; i < model.nodes_count; ++i) { // for each node

        const tg3_node& node = model.nodes[i];

        // each node may refer to a mesh or camera; we care about geometry only
        int32_t mesh_i = node.mesh;
        if (node.mesh == -1) {
            continue;
        }

        const tg3_mesh& mesh = model.meshes[mesh_i];

        for (int j = 0; j < mesh.primitives_count; ++j) { // for each primitive
            const tg3_primitive& prim = mesh.primitives[j];

            if (prim.indices == -1) continue;

            // get index info
            const tg3_accessor& indexAccessor = model.accessors[prim.indices];
            const tg3_buffer_view& indexViewBuf = model.buffer_views[indexAccessor.buffer_view];
            const tg3_buffer& indexBuf = model.buffers[indexViewBuf.buffer];

            uint64_t numIndices = indexAccessor.count;

            std::vector<uint32_t> indexData(numIndices);
            switch (indexAccessor.component_type) {
                case TG3_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const uint8_t* buf = reinterpret_cast<const uint8_t*>(indexBuf.data.data + indexViewBuf.byte_offset + indexAccessor.byte_offset);
                    for (int i = 0; i < numIndices; ++i) indexData[i] = buf[i];
                    break;
                }
                case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* buf = reinterpret_cast<const uint16_t*>(indexBuf.data.data + indexViewBuf.byte_offset + indexAccessor.byte_offset);
                    for (int i = 0; i < numIndices; ++i) indexData[i] = buf[i];
                    break;
                }
                case TG3_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t* buf = reinterpret_cast<const uint32_t*>(indexBuf.data.data + indexViewBuf.byte_offset + indexAccessor.byte_offset);
                    for (int i = 0; i < numIndices; ++i) indexData[i] = buf[i];
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported index type");
            }

            // get uv data
            std::vector<glm::vec2> primUVs;
            for (int k = 0; k < prim.attributes_count; ++k) {
                const tg3_str_int_pair& attr = prim.attributes[k];

                std::string attrib_name(attr.key.data, attr.key.len);
                int attr_i = attr.value;

                if (attrib_name == "TEXCOORD_0") {
                    const tg3_accessor& uvAccessor = model.accessors[attr_i];
                    const tg3_buffer_view& uvViewBuf = model.buffer_views[uvAccessor.buffer_view];
                    const tg3_buffer& uvBuf = model.buffers[uvViewBuf.buffer];

                    const glm::vec2* uvData = reinterpret_cast<const glm::vec2*>(uvBuf.data.data + uvViewBuf.byte_offset + uvAccessor.byte_offset);

                    for (int idx = 0; idx < uvAccessor.count; idx += 1) {
                        glm::vec2 uv = uvData[idx];

                        primUVs.push_back(uv);
                    }
                }
                else {
                    continue;
                }
            }

            // create materials 
            int primMaterialId = -1;
            if (modelData["MATERIAL"] != "none") { // if material for mesh already specified in the JSON
                primMaterialId = MatNameToID[modelData["MATERIAL"]];
            }
            else { // otherwise, build own material 
                Material m = {};


                if (prim.material != -1) { // if material specified in gltf
                    const tg3_material& mat = model.materials[prim.material];
                    // check transmission 
                    const tg3_extras_ext& ext = mat.ext;
                    uint32_t numExt = ext.extensions_count;

                    bool isTransmissive = false;
                    if (numExt > 0) {
                        const tg3_extension* allExt = ext.extensions;

                        // looking for 
                        const char* key = "KHR_materials_transmission";
                        size_t keyLen = strlen(key);

                        for (int n = 0; n < numExt; ++n) {
                            const tg3_extension& e = allExt[n];
                            if (e.name.len == keyLen && memcmp(e.name.data, key, keyLen) == 0) {
                                m.type = DIELECTRIC;
                                m.color = glm::vec3(0.0f);
                                m.refractionIndex = 1.5f;
                                m.absorption = glm::vec3(0.0f);
                                isTransmissive = true;
                                break;
                            }
                        }
                    }

                    if (isTransmissive) {
                        primMaterialId = (int)materials.size();
                        materials.emplace_back(m);
                        break;
                    }

                    // pre-populate with default opaque material 
                    m.type = METALLICWORKFLOW;
                    m.color = glm::vec3(1.0f);
                    m.metallic = 0.0f;
                    m.roughness = 1.0f;

                    const tg3_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;

                    // prepopulate metallic/roughness with multiplicative factor 
                    m.metallic = pbr.metallic_factor;
                    m.roughness = pbr.roughness_factor;

                    // load metallic roughness info
                    int roughIdx = pbr.metallic_roughness_texture.index;
                    if (roughIdx != -1) {
                        int src = model.textures[roughIdx].source;
                        auto it = idxToTex.find(src);
                        if (it != idxToTex.end()) {
                            m.texIdx = it->second;
                        }
                        else {
                            DecodedImage texRaw;
                            if (decodeImg(model, src, baseDir, texRaw)) {
                                m.roughmapIdx = (int)textures.size();
                                idxToTex[src] = m.roughmapIdx; 
                                textures.push_back(std::move(texRaw.pixels));
                                texDims.push_back(glm::vec2(texRaw.w, texRaw.h));
                                printf("loaded tex %d: %dx%d\n", m.roughmapIdx, texRaw.w, texRaw.h);
                            }
                            else {
                                printf("decode failed for image %d\n", src);
                            }
                        }
                    }
                    
                    // load base color info 
                    int texIdx = pbr.base_color_texture.index; 
                    if (texIdx != -1) { // if texture specified for base color 
                        int src = model.textures[texIdx].source;
                        if (src != -1) { // if image source specified 
                            auto it = idxToTex.find(src); // look for image key (already decoded?) 
                            if (it != idxToTex.end()) {
                                m.texIdx = it->second; // grab index
                            }
                            else { // otherwise, decode image
                                DecodedImage texRaw; 
                                if (decodeImg(model, src, baseDir, texRaw)) {
                                    m.texIdx = (int)textures.size(); // store index
                                    idxToTex[src] = m.texIdx; // log it 
                                    // store dimensions and pixel data to -> GPU later
                                    printf("pixels=%zu expected=%zu first texel=%d %d %d %d\n",
                                        texRaw.pixels.size(), (size_t)texRaw.w * texRaw.h * 4,
                                        texRaw.pixels[0], texRaw.pixels[1], texRaw.pixels[2], texRaw.pixels[3]);
                                    textures.push_back(std::move(texRaw.pixels));
                                    texDims.push_back(glm::vec2(texRaw.w, texRaw.h));
                                    printf("loaded tex %d: %dx%d\n", m.texIdx, texRaw.w, texRaw.h);
                                }
                                else {
                                    printf("decode failed for image %d\n", src);
                                }
                            }
                        }
                    }
                    else {
                        m.color = glm::vec3(pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2]);
                    }
                }

                primMaterialId = (int)materials.size(); // update material id
                materials.emplace_back(m);

            }
   
            // populate triangle data
            for (int k = 0; k < prim.attributes_count; ++k) {
                const tg3_str_int_pair& attr = prim.attributes[k];

                std::string attrib_name(attr.key.data, attr.key.len);
                int attr_i = attr.value;


                if (attrib_name == "POSITION") { // process position buffer

                    // get position info
                    const tg3_accessor& posAccessor = model.accessors[attr_i];
                    const tg3_buffer_view& posViewBuf = model.buffer_views[posAccessor.buffer_view];
                    const tg3_buffer& posBuf = model.buffers[posViewBuf.buffer];
                    // position coodinates are always floats 
                    const glm::vec3* positionData = reinterpret_cast<const glm::vec3*>(posBuf.data.data + posViewBuf.byte_offset + posAccessor.byte_offset);

                    // want to store triangle info that we can iterate through later to test intersections 

                    for (int idx = 0; idx < numIndices; idx += 3) {
                        // 3 indices = 1 triangle

                        Triangle tri;

                        // index data
                        uint32_t i0 = indexData[idx];
                        uint32_t i1 = indexData[idx + 1];
                        uint32_t i2 = indexData[idx + 2];

                        // position data 
                        glm::vec3 pos0 = positionData[i0];
                        glm::vec3 pos1 = positionData[i1];
                        glm::vec3 pos2 = positionData[i2];

                        // world space positions
                        tri.positions[0] = glm::vec3(transform * glm::vec4(pos0, 1.0f));
                        tri.positions[1] = glm::vec3(transform * glm::vec4(pos1, 1.0f));
                        tri.positions[2] = glm::vec3(transform * glm::vec4(pos2, 1.0f));

                        // calculate normal
                        glm::vec3 normal = glm::normalize(glm::cross(pos1 - pos0, pos2 - pos0));
                        tri.normal = glm::normalize(glm::vec3(inverseTranspose * glm::vec4(normal, 0.0f)));

                        // calculate centroid (in world space for BVH)
                        glm::vec3 centroid = (pos0 + pos1 + pos2) * 0.3333f;
                        tri.centroid = glm::vec3(transform * glm::vec4(centroid, 1.0f));

                        // get materialId from earlier 
                        tri.materialid = primMaterialId;
                        
                        // uvs 
                        if (!primUVs.empty()) {
                            glm::vec2 uv0 = primUVs[i0];
                            glm::vec2 uv1 = primUVs[i1];
                            glm::vec2 uv2 = primUVs[i2];

                            tri.uv[0] = uv0;
                            tri.uv[1] = uv1;
                            tri.uv[2] = uv2;
                        }

                        triangles.push_back(tri);
                    }
                }
                else { // don't care about other buffers atm
                    continue;
                }
            }
        }
    }
    
    // free resources
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
}

