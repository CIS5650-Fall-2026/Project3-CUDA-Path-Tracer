#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "glm/glm.hpp"

#include <tiny_obj_loader.h>
#include <tiny_gltf_v3.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>
#include <stb_image.h>

using namespace std;
using json = nlohmann::json;

static glm::ivec3 storeTexture(
    unsigned char* image, int width, int height, 
    std::vector<uchar4>& texPixels
) {
    if (image == nullptr) {
        return glm::ivec3(0);
    }

    int offset = texPixels.size();
    texPixels.resize(offset + width * height);
    for (int i = 0; i < width * height; ++i) {
        texPixels[offset + i] = make_uchar4(
            image[4 * i],
            image[4 * i + 1],
            image[4 * i + 2],
            image[4 * i + 3]
        );
    }

    stbi_image_free(image);
    return glm::ivec3(offset, width, height);
}

static glm::ivec3 loadTextureFile(const std::string &filename, std::vector<uchar4> &texPixels) {
    int w, h, channels;
    unsigned char *image = stbi_load(filename.c_str(), &w, &h, &channels, 4);
    if (image == nullptr) {
        std::cerr << "Error loading texture " << filename << ": " << stbi_failure_reason() << std::endl;
        return glm::ivec3(0);
    }
    return storeTexture(image, w, h, texPixels);
}

static glm::ivec3 loadTextureMemory(const uint8_t *data, int size, std::vector<uchar4> &texPixels) {
    int w, h, channels;
    unsigned char *image = stbi_load_from_memory(data, size, &w, &h, &channels, 4);
    if (image == nullptr) {
        std::cerr << "Error loading texture: " << stbi_failure_reason() << std::endl;
        return glm::ivec3(0);
    }
    return storeTexture(image, w, h, texPixels);
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        loadFromJSON(filename);
        return;
    } else if (ext == ".obj") {
        loadFromOBJ(filename);
    } else if (ext == ".gltf" || ext == ".glb") {
        loadFromGLTF(filename);
    } else {
        cout << "Couldn't read from " << filename << endl;
        return;
    }

    appendMeshToRender();
    initMeshCamera();
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
        else if (p["TYPE"] == "Refractive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1;
            newMaterial.indexOfRefraction = p["IOR"];
        }

        // load texture
        const std::string texture = p.value("TEXTURE", std::string("Solid"));
        if (texture == "Checker")
        {
            newMaterial.textureType = CHECKER;
        }
        else if (texture == "Stripes")
        {
            newMaterial.textureType = STRIPES;
        } else
        {
            newMaterial.textureType = SOLID;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
        glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        glm::vec3 geomScale = glm::vec3(scale[0], scale[1], scale[2]);
        glm::mat4 transform = utilityCore::buildTransformationMatrix(
            translation, rotation, geomScale);
        int materialId = MatNameToID[p["MATERIAL"]];

        // check if need to procedurally generate first
        if (type == "shell") {
            generateShell(materialId, transform);
            continue;
        } else if (type == "vase") {
            generateVase(materialId, transform);
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
        newGeom.materialid = materialId;
        newGeom.translation = translation;
        newGeom.rotation = rotation;
        newGeom.scale = geomScale;
        newGeom.transform = transform;
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

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0);
    camera.focalDistance = cameraData.value(
        "FOCAL_DISTANCE", glm::length(camera.lookAt - camera.position));

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromGLTF(const std::string& gltfName) {
    tg3_model model{};
    tg3_error_stack errors{};
    tg3_parse_options options{};

    tg3_error_stack_init(&errors);
    tg3_parse_options_init(&options);
    
    const tg3_error_code loaded = tg3_parse_file(
        &model, &errors, gltfName.c_str(), gltfName.length(), &options);
    
    if (loaded != TG3_OK) {
        // error handle
        std::string error_msg = "Error loading glTF: \n";
        for (int i = 0; i < errors.count; ++i) {
            if (errors.entries[i].message) {
                error_msg += "Error " + std::to_string(i) + ": " 
                + errors.entries[i].message + '\n';
            }
        }

        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        
        throw std::runtime_error(error_msg);
    }

    mesh = {};

    // load images embedded into gltf
    std::vector<glm::ivec3> gltfImages(model.images_count, glm::ivec3(0));
    std::filesystem::path baseDirectory = std::filesystem::path(gltfName).parent_path();

    for (int i = 0; i < model.images_count; ++i) {
        const tg3_image& image = model.images[i];

        if (image.buffer_view >= 0) {
            const tg3_buffer_view& view = model.buffer_views[image.buffer_view];
            const tg3_buffer& buffer = model.buffers[view.buffer];

            gltfImages[i] = loadTextureMemory(
                buffer.data.data + view.byte_offset, view.byte_length, texPixels
            );
        } else if (image.uri.len > 0) {
            std::string uri(image.uri.data, image.uri.len);

            if (uri.rfind("data:", 0) == 0) {
                std::cerr << "Error loading gltf texture" << std::endl;
            } else {
                gltfImages[i] = loadTextureFile((baseDirectory / uri).string(), texPixels);
            }
        }
    }

    const int default_material_id = materials.size();
    Material default_material{};
    default_material.color = glm::vec3(1);
    materials.push_back(default_material);

    const int material_offset = materials.size();

    // load materials
    for (int i = 0; i < model.materials_count; ++i) {
        Material curr_material{};

        curr_material.color = glm::vec3(
            model.materials[i].pbr_metallic_roughness.base_color_factor[0],
            model.materials[i].pbr_metallic_roughness.base_color_factor[1],
            model.materials[i].pbr_metallic_roughness.base_color_factor[2]
        );
        
        curr_material.emittance = glm::max(
            model.materials[i].emissive_factor[0], glm::max(
                model.materials[i].emissive_factor[1],
                model.materials[i].emissive_factor[2]
            ));

        int texIndex = model.materials[i].pbr_metallic_roughness.base_color_texture.index;
        if (texIndex >= 0 && texIndex < model.textures_count) {
            int imageIndex = model.textures[texIndex].source;
            if (imageIndex >= 0 && imageIndex < gltfImages.size()) {
                curr_material.colorTexOffset = gltfImages[imageIndex].x;
                curr_material.colorTexWidth = gltfImages[imageIndex].y;
                curr_material.colorTexHeight = gltfImages[imageIndex].z;
            }
        }
        materials.push_back(curr_material);
    }

    for (int i = 0; i < model.meshes_count; ++i) {
        const tg3_mesh curr_mesh = model.meshes[i];
        
        for (int j = 0; j < curr_mesh.primitives_count; ++j) {
            const tg3_primitive &primitive = curr_mesh.primitives[j];
            // load triangles
            if (primitive.mode != TG3_MODE_TRIANGLES) {
                continue;
            }

            const int pos_accessor = findAttribute(primitive, "POSITION");
            if (pos_accessor < 0) {
                continue;
            }

            const int vertex_offset = appendFloatAttribute(
                model, pos_accessor, 3, mesh.vertices);
            if (vertex_offset < 0) {
                continue;
            }

            const int normal_offset = appendFloatAttribute(
                model,
                findAttribute(primitive, "NORMAL"),
                3,
                mesh.normals
            );

            const int texcoord_offset = appendFloatAttribute(
                model,
                findAttribute(primitive, "TEXCOORD_0"),
                2,
                mesh.texcoords
            );

            const unsigned long index_count = appendGLTFIndices(
                model, primitive, vertex_offset, normal_offset, texcoord_offset);

            const int material_id = primitive.material < 0 
                ? default_material_id : material_offset + primitive.material;
        
            for (int i = 0; i < index_count / 3; ++i) {
                mesh.num_face_vertices.push_back(3);
                mesh.material_ids.push_back(material_id);
                mesh.smoothing_group_ids.push_back(0);
            }
        }
    }

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
    
    // success!!
    std::cout << "Success! Loaded GLTF" << std::endl;
}

void Scene::loadFromOBJ(const std::string& objName){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn;
    std::string err;

    std::filesystem::path objPath(objName);
    std::string baseDirectory = objPath.parent_path().string() 
        + std::filesystem::path::preferred_separator;
    
    bool loaded = LoadObj(
        &attrib, &shapes, &objMaterials, &warn, &err, 
        objName.c_str(), baseDirectory.c_str());

    if (!warn.empty()) {
        std::cerr << "Warning: " << warn << std::endl;
    }
    if (!loaded) {
        throw std::runtime_error("Error loading OBJ: " + err);
    }

    mesh.vertices.assign(attrib.vertices.begin(), attrib.vertices.end());
    mesh.normals.assign(attrib.normals.begin(), attrib.normals.end());
    mesh.texcoords.assign(attrib.texcoords.begin(), attrib.texcoords.end());
    // need to flip obj v coords
    for (size_t i = 1; i < mesh.texcoords.size(); i += 2) {
        mesh.texcoords[i] = 1 - mesh.texcoords[i];
    }

    // set default material for obj
    Material defaultMaterial{};
    defaultMaterial.color = glm::vec3(1.0f);

    const int defaultMaterialId = this->materials.size();

    this->materials.push_back(defaultMaterial);

    // should be 1
    const int objMaterialOffset = this->materials.size();

    // actually set the right material
    for (const tinyobj::material_t &objMaterial : objMaterials) {
        Material material{};

        const glm::vec3 diffuse(
            objMaterial.diffuse[0],
            objMaterial.diffuse[1],
            objMaterial.diffuse[2]
        );

        const glm::vec3 emission(
            objMaterial.emission[0],
            objMaterial.emission[1],
            objMaterial.emission[2]
        );

        material.emittance = glm::max(
            emission.x, glm::max(emission.y, emission.z)
        );

        // for emissixe materials, color is normalized emission color
        // o.w. storing diffuse color
        if (material.emittance > 0) {
            material.color = emission / material.emittance;
        } else {
            material.color = diffuse;
        }

        // load textures
        if (!objMaterial.diffuse_texname.empty()) {
            std::string filename = (std::filesystem::path(baseDirectory) / objMaterial.diffuse_texname).string();
            glm::ivec3 texture = loadTextureFile(filename, texPixels);
            material.colorTexOffset = texture.x;
            material.colorTexWidth = texture.y;
            material.colorTexHeight = texture.z;
        }

        if (!objMaterial.bump_texname.empty()) {
            std::string filename = (std::filesystem::path(baseDirectory) / objMaterial.bump_texname).string();
            glm::ivec3 texture = loadTextureFile(filename, texPixels);
            material.bumpTexOffset = texture.x;
            material.bumpTexWidth = texture.y;
            material.bumpTexHeight = texture.z;
            material.bumpStrength = 0.02f * objMaterial.bump_texopt.bump_multiplier;
        }
        this->materials.push_back(material);
    }
    

    for (const tinyobj::shape_t &s : shapes) {
        for (const tinyobj::index_t &i : s.mesh.indices) {
            mesh.indices.push_back(
                {i.vertex_index, i.normal_index, i.texcoord_index}
            );
        }

        mesh.num_face_vertices.insert(
            mesh.num_face_vertices.end(), 
            s.mesh.num_face_vertices.begin(),
            s.mesh.num_face_vertices.end()
        );
            
        for (size_t face = 0; face < s.mesh.num_face_vertices.size(); ++face) {
            const int objMaterialId = face < s.mesh.material_ids.size() ? s.mesh.material_ids[face] : -1;

            if (objMaterialId >= 0 && objMaterialId < objMaterials.size()) {
                mesh.material_ids.push_back(objMaterialOffset + objMaterialId);
            } else {
                mesh.material_ids.push_back(defaultMaterialId);
            }
        }

        mesh.smoothing_group_ids.insert(
            mesh.smoothing_group_ids.end(),
            s.mesh.smoothing_group_ids.begin(),
            s.mesh.smoothing_group_ids.end()
        );
    }

    

    // success!!
    std::cout << "Success! Loaded OBJ" << std::endl;
}

void Scene::appendMeshToRender() {
    size_t offset = 0;
    for (std::size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
        if (mesh.num_face_vertices[f] != 3) {
            // not a triangle, should not get here
            offset += mesh.num_face_vertices[f];
            continue;
        }

        Geom triangle{};
        triangle.type = TRIANGLE;
        triangle.materialid = mesh.material_ids[f];
        triangle.hasNormals = true;
        triangle.hasUVs = true;

        for (int j = 0; j < 3; ++j) {
            const index_t &index = mesh.indices[offset + j];

            triangle.triangleVertices[j] = glm::vec3(
                mesh.vertices[3 * index.vertex_index],
                mesh.vertices[3 * index.vertex_index + 1],
                mesh.vertices[3 * index.vertex_index + 2]
            );

            if (index.normal_index < 0) {
                triangle.hasNormals = false;
            } else {
                triangle.triangleNormals[j] = glm::vec3(
                    mesh.normals[3 * index.normal_index],
                    mesh.normals[3 * index.normal_index + 1],
                    mesh.normals[3 * index.normal_index + 2]
                );
            }

            if (index.texcoord_index < 0) {
                triangle.hasUVs = false;
            } else {
                triangle.triangleUVs[j] = glm::vec2(
                    mesh.texcoords[2 * index.texcoord_index],
                    mesh.texcoords[3 * index.texcoord_index + 1]
                );
            }
        }

        geoms.push_back(triangle);
        offset += 3;
    }
}

int Scene::findAttribute(const tg3_primitive &primitive, const char *name) {
    for (int i = 0; i < primitive.attributes_count; ++i) {
        if (tg3_str_equals_cstr(primitive.attributes[i].key, name)) {
            return primitive.attributes[i].value;
        }
    }

    return -1;
}
    
const uint8_t* Scene::getAccessorData(
    const tg3_model &model,
    const tg3_accessor &accessor,
    int32_t &stride
) {
    const tg3_buffer_view &view = model.buffer_views[accessor.buffer_view];
    const tg3_buffer &buffer = model.buffers[view.buffer];
    stride = tg3_accessor_byte_stride(&accessor, &view);
    
    return buffer.data.data + view.byte_offset + accessor.byte_offset;
}

int Scene::appendFloatAttribute(
    const tg3_model &model,
    int accessorIdx,
    int componentCount,
    std::vector<float> &dest
) {
    if (accessorIdx < 0) {
        return -1;
    }

    const tg3_accessor &accessor = model.accessors[accessorIdx];

    if (accessor.component_type != TG3_COMPONENT_TYPE_FLOAT) {
        return -1;
    }

    const int offset = dest.size() / componentCount;
    int32_t stride;
    const uint8_t *data = getAccessorData(model, accessor, stride);

    for (int i = 0; i < accessor.count; ++i) {
        const float *val = (float*) (data + i * stride);

        for (int component = 0; component < componentCount; ++component) {
            dest.push_back(val[component]);
        }
    }

    return offset;
}

unsigned long int Scene::appendGLTFIndices(
    const tg3_model &model,
    const tg3_primitive &primitive,
    int vertex_offset,
    int normal_offset,
    int texcoord_offset
) {
    const tg3_accessor &pos = model.accessors[findAttribute(primitive, "POSITION")];
    const tg3_accessor *index_accessor = nullptr;
    const uint8_t *index_data = nullptr;
    int index_stride = 0;
    uint64_t index_count = pos.count;

    if (primitive.indices >= 0) {
        index_accessor = &model.accessors[primitive.indices];
        index_data = getAccessorData(model, *index_accessor, index_stride);
        index_count = index_accessor->count;
    }

    for (int i = 0; i < index_count; ++i) {
        int idx = i;
        if (index_accessor) {
            const uint8_t *value = index_data + i * index_stride;

            switch (index_accessor->component_type) {
                case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
                    idx = *(uint8_t*)(value);
                    break;
                case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
                    idx = *(uint16_t*)(value);
                    break;
                case TG3_COMPONENT_TYPE_UNSIGNED_INT:
                    idx = *(uint32_t*)(value);
                    break;
                default:
                    throw std::runtime_error("Wrong type for glTF index");
            }
        }
            
        mesh.indices.push_back({
            vertex_offset + idx,
            normal_offset < 0 ? -1 : normal_offset + idx,
            texcoord_offset < 0 ? -1 : texcoord_offset + idx
        });
    }

    return index_count;
}

void Scene::initMeshCamera() {
    if (geoms.empty()) {
        throw std::runtime_error("Mesh contains no triangles");
    }

    glm::vec3 boundsMin(FLT_MAX);
    glm::vec3 boundsMax(-FLT_MAX);

    for (const Geom& geom : geoms) {
        if (geom.type != TRIANGLE) {
            continue;
        }
        for (int i = 0; i < 3; ++i) {
            boundsMin = glm::min(boundsMin, geom.triangleVertices[i]);
            boundsMax = glm::max(boundsMax, geom.triangleVertices[i]);
        }
    }

    const glm::vec3 center = 0.5f * (boundsMin + boundsMax);
    const float radius = glm::max(0.5f * glm::length(boundsMax - boundsMin), 0.001f);

    Camera& camera = state.camera;
    camera.resolution = glm::ivec2(800, 800);

    // setting perspective slightly elevated.
    const glm::vec3 viewDirection = glm::normalize(glm::vec3(1, 0.35, 1));

    const float halfFovY = glm::radians(45.0f);
    const float distance = 1.15 * radius / sin(halfFovY);

    camera.lookAt = center;
    camera.position = center + viewDirection * distance;
    camera.up = glm::vec3(0, 1, 0);
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    // default pinhole camera for obj and gltf files
    camera.lensRadius = 0;
    camera.focalDistance = glm::length(camera.lookAt - camera.position);

    const float yScale = tan(halfFovY);
    const float aspect = (float)(camera.resolution.x) / camera.resolution.y;
    const float xScale = yScale * aspect;

    camera.fov = glm::vec2(glm::degrees(atan(xScale)), glm::degrees(halfFovY));

    camera.pixelLength = glm::vec2(
        2 * xScale / camera.resolution.x, 2 * yScale / camera.resolution.y);

    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = "mesh";
    state.image.assign(camera.resolution.x * camera.resolution.y, glm::vec3());
}

// params for procedural shells
#define TOTAL_ANGLE 8 * PI
#define GROWTH_EXP_FACTOR 0.15
#define CENTER_RADIUS_FACTOR 2
#define APERTURE_RADIUS_FACTOR 0.5
#define U_SEGMENTS 96
#define V_SEGMENTS 24

glm::vec3 Scene::shellPoint(float u, float v) {
    float growth = expf(GROWTH_EXP_FACTOR * (u - TOTAL_ANGLE));
    float centerRadius = CENTER_RADIUS_FACTOR * growth;
    float apertureRadius = APERTURE_RADIUS_FACTOR * growth;
    float radialDistance = centerRadius + apertureRadius * cosf(v);

    return glm::vec3(
        radialDistance * cosf(u), growth + 0.8 * apertureRadius * sinf(v), 
        radialDistance * sinf(u));
}

glm::vec3 Scene::vasePoint(float u, float v) {
    float height = 3 * u;
    float profile = 0.75 + 0.25 * sinf(PI * u);
    float waves = 1 + 0.15 * sinf(8 * v + 4* PI * u);
    float radialDistance = profile * waves;

    return glm::vec3(radialDistance * cosf(v), height, radialDistance * sinf(v));
}

void Scene::addProceduralTriangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, int materialId) {
    Geom t{};
    t.type = TRIANGLE;
    t.materialid = materialId;
    t.hasNormals = false;
    t.triangleVertices[0] = a;
    t.triangleVertices[1] = b;
    t.triangleVertices[2] = c;

    geoms.push_back(t);
}

void Scene::generateShell(int materialId, const glm::mat4& transform) {
    for (int i = 0; i < U_SEGMENTS; ++i) {
        float u0 = TOTAL_ANGLE * i / U_SEGMENTS;
        float u1 = TOTAL_ANGLE * (i + 1) / U_SEGMENTS;

        for (int j = 0; j < V_SEGMENTS; ++j) {
            float v0 = 2 * PI * j / V_SEGMENTS;
            float v1 = 2 * PI * (j + 1) / V_SEGMENTS;

            glm::vec3 p00 = glm::vec3(transform * glm::vec4(shellPoint(u0, v0), 1.0f));
            glm::vec3 p01 = glm::vec3(transform * glm::vec4(shellPoint(u0, v1), 1.0f));
            glm::vec3 p10 = glm::vec3(transform * glm::vec4(shellPoint(u1, v0), 1.0f));
            glm::vec3 p11 = glm::vec3(transform * glm::vec4(shellPoint(u1, v1), 1.0f));

            addProceduralTriangle(p00, p01, p11, materialId);
            addProceduralTriangle(p00, p11, p10, materialId);
        }
    }
}

void Scene::generateVase(int materialId, const glm::mat4& transform) {
    for (int i = 0; i < U_SEGMENTS; ++i) {
        float u0 = i / U_SEGMENTS;
        float u1 = (i + 1) / U_SEGMENTS;

        for (int j = 0; j < V_SEGMENTS; ++j) {
            float v0 = 2 * PI * j / V_SEGMENTS;
            float v1 = 2 * PI * (j + 1) / V_SEGMENTS;

            glm::vec3 p00 = glm::vec3(transform * glm::vec4(vasePoint(u0, v0), 1.0f));
            glm::vec3 p01 = glm::vec3(transform * glm::vec4(vasePoint(u0, v1), 1.0f));
            glm::vec3 p10 = glm::vec3(transform * glm::vec4(vasePoint(u1, v0), 1.0f));
            glm::vec3 p11 = glm::vec3(transform * glm::vec4(vasePoint(u1, v1), 1.0f));

            addProceduralTriangle(p00, p01, p11, materialId);
            addProceduralTriangle(p00, p11, p10, materialId);
        }
    }
}