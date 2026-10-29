// glTF loading through tinygltf; the one TU that compiles tinygltf's implementation, on the repo's json.hpp.
// Images decode through stb_image to RGBA, 8 or 16 bits; a missing image file only warns, an undecodable one fails the load.

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "json.hpp"
#include "tiny_gltf.h"

#include "scene/gltf_loader.h"
#include "scene/scene.h"
#include "timing.h"
#include "utilities.h"

#include "mikktspace.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
// Photometric constant for KHR_lights_punctual's candela and lux.
constexpr double LUMENS_PER_WATT = 683.0;

// One accessor's elements; buffer views may interleave attributes, so element i sits at data + i * stride.
struct AccessorView
{
    const unsigned char* data = nullptr;
    size_t stride = 0;
    size_t count = 0;
    int componentType = 0;   // TINYGLTF_COMPONENT_TYPE_*
    int numComponents = 0;   // 1 for SCALAR, 3 for VEC3, ...

    const unsigned char* at(size_t i) const { return data + i * stride; }
};

bool viewAccessor(const tinygltf::Model& model, int accessorIndex, AccessorView& view, std::string& err)
{
    if (accessorIndex < 0 || accessorIndex >= (int)model.accessors.size())
    {
        err = "accessor index out of range";
        return false;
    }
    const tinygltf::Accessor& a = model.accessors[accessorIndex];
    if (a.bufferView < 0 || a.bufferView >= (int)model.bufferViews.size())
    {
        err = "accessor without a buffer view (sparse, Draco or meshopt data is not supported)";
        return false;
    }
    if (a.sparse.isSparse)
    {
        err = "sparse accessors are not supported";
        return false;
    }
    const tinygltf::BufferView& bv = model.bufferViews[a.bufferView];
    // tinygltf checks this index for images only, not for mesh data.
    if (bv.buffer < 0 || bv.buffer >= (int)model.buffers.size())
    {
        err = "buffer view with a bad buffer index";
        return false;
    }
    const tinygltf::Buffer& buffer = model.buffers[bv.buffer];
    const int stride = a.ByteStride(bv);  // the view's byteStride, or the packed element size
    const int componentSize = tinygltf::GetComponentSizeInBytes(a.componentType);
    const int numComponents = tinygltf::GetNumComponentsInType(a.type);
    if (stride <= 0 || componentSize <= 0 || numComponents <= 0)
    {
        err = "accessor with an unsupported layout";
        return false;
    }
    // count is bounded by the buffer size so the product cannot wrap.
    const size_t begin = bv.byteOffset + a.byteOffset;
    const size_t end = a.count == 0 ? begin : begin + (a.count - 1) * stride + componentSize * numComponents;
    if (a.count > buffer.data.size() || end > buffer.data.size())
    {
        err = "accessor reaches past the end of its buffer";
        return false;
    }
    view.data = buffer.data.data() + begin;
    view.stride = stride;
    view.count = a.count;
    view.componentType = a.componentType;
    view.numComponents = numComponents;
    return true;
}

bool isFloatVec3(const AccessorView& v)
{
    return v.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && v.numComponents == 3;
}

glm::vec3 readVec3(const AccessorView& v, size_t i)
{
    const float* f = reinterpret_cast<const float*>(v.at(i));
    return glm::vec3(f[0], f[1], f[2]);
}

bool isIndexType(int componentType)
{
    return componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
}

unsigned int readIndex(const AccessorView& v, size_t i)
{
    const unsigned char* p = v.at(i);
    switch (v.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return *p;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return *reinterpret_cast<const uint16_t*>(p);
    default:
        return *reinterpret_cast<const uint32_t*>(p);
    }
}

// Integer texture coordinates are normalized to [0, 1] per the glTF spec.
bool isUvType(int componentType)
{
    return componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
}

glm::vec2 readUv(const AccessorView& v, size_t i)
{
    const unsigned char* p = v.at(i);
    switch (v.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return glm::vec2(p[0], p[1]) / 255.0f;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(p);
        return glm::vec2(s[0], s[1]) / 65535.0f;
    }
    default:
    {
        const float* f = reinterpret_cast<const float*>(p);
        return glm::vec2(f[0], f[1]);
    }
    }
}

cudaTextureAddressMode addressMode(int gltfWrap)
{
    switch (gltfWrap)
    {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
        return cudaAddressModeClamp;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
        return cudaAddressModeMirror;
    default:
        return cudaAddressModeWrap;
    }
}

glm::mat4 nodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        // glTF stores matrices column-major, the same as glm.
        glm::mat4 m;
        for (int i = 0; i < 16; ++i)
        {
            m[i / 4][i % 4] = (float)node.matrix[i];
        }
        return m;
    }
    glm::mat4 t(1.0f);
    glm::mat4 r(1.0f);
    glm::mat4 s(1.0f);
    if (node.translation.size() == 3)
    {
        t = glm::translate(glm::mat4(1.0f),
            glm::vec3((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]));
    }
    if (node.rotation.size() == 4)
    {
        // glTF quaternions are (x, y, z, w); glm's constructor takes w first.
        r = glm::mat4_cast(glm::quat((float)node.rotation[3],
            (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
    }
    if (node.scale.size() == 3)
    {
        s = glm::scale(glm::mat4(1.0f),
            glm::vec3((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]));
    }
    return t * r * s;
}

// object[key], or null when absent; Value::Get asserts on non-objects.
const tinygltf::Value& member(const tinygltf::Value& object, const char* key)
{
    static const tinygltf::Value null;
    return object.Has(key) ? object.Get(key) : null;
}

float numberMember(const tinygltf::Value& object, const char* key, float fallback)
{
    const tinygltf::Value& value = member(object, key);
    return value.IsNumber() ? (float)value.GetNumberAsDouble() : fallback;
}

const tinygltf::Value& extension(const tinygltf::Material& material, const char* name)
{
    static const tinygltf::Value null;
    auto found = material.extensions.find(name);
    return found == material.extensions.end() ? null : found->second;
}

float extensionNumber(const tinygltf::Material& material, const char* name, const char* key, float fallback)
{
    return numberMember(extension(material, name), key, fallback);
}

glm::vec3 extensionColor(const tinygltf::Material& material, const char* name, const char* key, glm::vec3 fallback)
{
    const tinygltf::Value& value = member(extension(material, name), key);
    if (value.ArrayLen() != 3)
    {
        return fallback;
    }
    glm::vec3 color;
    for (int i = 0; i < 3; ++i)
    {
        const tinygltf::Value& c = value.Get(i);
        if (!c.IsNumber())
        {
            return fallback;
        }
        color[i] = (float)c.GetNumberAsDouble();
    }
    return color;
}

// An extension's texture slot as a TextureInfo; index -1 when there is none.
tinygltf::TextureInfo extensionTexture(const tinygltf::Material& material, const char* name, const char* key)
{
    const tinygltf::Value& slot = member(extension(material, name), key);
    tinygltf::TextureInfo info;
    info.index = (int)numberMember(slot, "index", -1.0f);
    info.texCoord = (int)numberMember(slot, "texCoord", 0.0f);
    return info;
}

const char* const READ_EXTENSIONS[] = {
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_specular",
    "KHR_materials_clearcoat",
};

bool isReadExtension(const std::string& name)
{
    for (const char* read : READ_EXTENSIONS)
    {
        if (name == read)
        {
            return true;
        }
    }
    return false;
}

// Prints each unsupported material extension once per file; occlusion is not named, path tracing computes it.
void reportUnsupportedMaterialFeatures(const tinygltf::Model& model, const std::string& path)
{
    std::map<std::string, int> extensions;
    for (const tinygltf::Material& material : model.materials)
    {
        std::set<std::string> unread;
        for (const auto& e : material.extensions)
        {
            if (!isReadExtension(e.first))
            {
                unread.insert(e.first);
            }
        }
        // Texture-level extensions (KHR_texture_transform) are not applied.
        const tinygltf::ExtensionMap* slots[] = {
            &material.pbrMetallicRoughness.baseColorTexture.extensions,
            &material.pbrMetallicRoughness.metallicRoughnessTexture.extensions,
            &material.normalTexture.extensions,
            &material.emissiveTexture.extensions,
        };
        for (const tinygltf::ExtensionMap* slot : slots)
        {
            for (const auto& e : *slot)
            {
                unread.insert(e.first);
            }
        }
        const tinygltf::Value& transmissionSlot =
            member(member(extension(material, "KHR_materials_transmission"), "transmissionTexture"), "extensions");
        if (transmissionSlot.IsObject())
        {
            for (const std::string& key : transmissionSlot.Keys())
            {
                unread.insert(key);
            }
        }
        for (const std::string& name : unread)
        {
            ++extensions[name];
        }
    }
    for (const auto& e : extensions)
    {
        fprintf(stderr, "glTF %s: %s ignored (%d materials)\n", path.c_str(), e.first.c_str(), e.second);
    }
}

// fov_degrees is skipped (yfov already holds it) and samples_per_pixel too (tuned for PBRT).
GltfRenderHints renderHints(const tinygltf::Model& model)
{
    GltfRenderHints hints;
    const tinygltf::Value& render = member(member(model.extras, "pbrt"), "render");
    const tinygltf::Value& resolution = member(render, "resolution");
    if (resolution.ArrayLen() == 2)
    {
        hints.width = resolution.Get(0).GetNumberAsInt();
        hints.height = resolution.Get(1).GetNumberAsInt();
    }
    hints.maxDepth = (int)numberMember(render, "max_depth", 0.0f);
    return hints;
}

// Light linking from node extras lightMask/lightGroup, as scenes/blender-migration/tools/export_stardust.py writes them.
// The exporter keeps groups to 0..30 so a mask fits a JSON integer; the low 32 bits cover a negative parse.
struct LightLink
{
    unsigned int mask = 0xffffffffu;
    int group = 0;
};

LightLink lightLinkFor(const tinygltf::Node& node, const std::string& path)
{
    LightLink link;
    const tinygltf::Value& mask = member(node.extras, "lightMask");
    if (mask.IsNumber())
    {
        link.mask = (unsigned int)(long long)mask.GetNumberAsDouble() | 1u;
    }
    const tinygltf::Value& group = member(node.extras, "lightGroup");
    if (group.IsNumber())
    {
        const int g = (int)group.GetNumberAsDouble();
        if (g < 0 || g > 31)
        {
            fprintf(stderr, "glTF %s: node \"%s\" has lightGroup %d, outside 0..31; left in group 0\n", path.c_str(),
                node.name.c_str(), g);
        }
        else
        {
            link.group = g;
        }
    }
    return link;
}

bool isInfiniteLight(const tinygltf::Value& light)
{
    const tinygltf::Value& kind = member(light, "pbrt");
    return kind.IsString() && kind.Get<std::string>() == "infinite";
}

// "source" is relative to the PBRT scene folder, which the research scenes keep as source/ next to gltf/.
std::optional<GltfEnvironment> pbrtEnvironment(const tinygltf::Model& model, const std::string& path)
{
    const tinygltf::Value& sources = member(member(model.extras, "pbrt"), "light_sources");
    for (size_t i = 0; i < sources.ArrayLen(); ++i)
    {
        const tinygltf::Value& light = sources.Get(i);
        if (!isInfiniteLight(light))
        {
            continue;
        }
        GltfEnvironment env;
        env.radiance = glm::vec3(1.0f);
        const tinygltf::Value& source = member(light, "source");
        if (source.IsString())
        {
            const std::filesystem::path pbrtFile =
                std::filesystem::path(path).parent_path().parent_path() / "source" / source.Get<std::string>();
            env.pbrtFile = pbrtFile.generic_string();
            env.latlongFile = std::filesystem::path(pbrtFile).replace_extension(".latlong.exr").generic_string();
        }
        else
        {
            const tinygltf::Value& rgb = member(light, "radiance_rgb");
            if (rgb.ArrayLen() == 3)
            {
                env.radiance = glm::vec3(rgb.Get(0).GetNumberAsDouble(), rgb.Get(1).GetNumberAsDouble(),
                    rgb.Get(2).GetNumberAsDouble());
            }
        }
        return env;
    }
    return std::nullopt;
}

bool isDistantLight(const tinygltf::Value& light)
{
    const tinygltf::Value& kind = member(light, "pbrt");
    return kind.IsString() && kind.Get<std::string>() == "distant";
}

// radiance_rgb is irradiance facing the light; pbrt_transform is column-major, in PBRT's world coordinates.
void addPbrtDistantLights(const tinygltf::Model& model, Scene& scene)
{
    const tinygltf::Value& sources = member(member(model.extras, "pbrt"), "light_sources");
    auto vec3Member = [](const tinygltf::Value& object, const char* key, glm::vec3 fallback) {
        const tinygltf::Value& v = member(object, key);
        return v.ArrayLen() == 3 ? glm::vec3(v.Get(0).GetNumberAsDouble(), v.Get(1).GetNumberAsDouble(),
                                       v.Get(2).GetNumberAsDouble())
                                 : fallback;
    };
    for (size_t i = 0; i < sources.ArrayLen(); ++i)
    {
        const tinygltf::Value& light = sources.Get(i);
        if (!isDistantLight(light))
        {
            continue;
        }
        glm::mat4 worldFromLight(1.0f);
        const tinygltf::Value& transform = member(light, "pbrt_transform");
        if (transform.ArrayLen() == 16)
        {
            for (int k = 0; k < 16; ++k)
            {
                worldFromLight[k / 4][k % 4] = (float)transform.Get(k).GetNumberAsDouble();
            }
        }
        const glm::vec3 toLight = vec3Member(light, "from", glm::vec3(0.0f, 0.0f, 0.0f))
                                - vec3Member(light, "to", glm::vec3(0.0f, 0.0f, 1.0f));
        PunctualLight p{};
        p.type = LIGHT_DISTANT;
        p.position = glm::normalize(glm::mat3(worldFromLight) * toLight);
        p.intensity = vec3Member(light, "radiance_rgb", glm::vec3(1.0f));
        if (maxComponent(p.intensity) > 0.0f)
        {
            scene.punctualLights.push_back(p);
        }
    }
}

// Prints the file's lights that do not reach the render; research scenes mark PBRT-only lights "metadata_only".
void reportUnusedLights(const tinygltf::Model& model, const std::string& path, bool sceneLightsRead)
{
    int dark = 0;
    int ranged = 0;
    for (const tinygltf::Light& light : model.lights)
    {
        dark += light.intensity <= 0.0;
        ranged += light.intensity > 0.0 && light.range > 0.0;
    }
    if (dark > 0)
    {
        fprintf(stderr, "glTF %s: %d punctual lights have intensity 0 and add no light\n", path.c_str(), dark);
    }
    if (ranged > 0)
    {
        fprintf(stderr, "glTF %s: range of %d punctual lights ignored, they light without a cutoff\n", path.c_str(),
            ranged);
    }
    const tinygltf::Value& sources = member(member(model.extras, "pbrt"), "light_sources");
    for (size_t i = 0; i < sources.ArrayLen(); ++i)
    {
        const tinygltf::Value& light = sources.Get(i);
        const tinygltf::Value& kind = member(light, "pbrt");
        const tinygltf::Value& form = member(light, "gltf");
        if (!form.IsString() || form.Get<std::string>() != "metadata_only")
        {
            continue;
        }
        if (isInfiniteLight(light) || isDistantLight(light))
        {
            if (!sceneLightsRead)
            {
                fprintf(stderr, "glTF %s: PBRT %s light ignored (only read when the file is the whole scene)\n",
                    path.c_str(), kind.Get<std::string>().c_str());
            }
            continue;
        }
        fprintf(stderr, "glTF %s: PBRT %s light ignored (not supported)\n", path.c_str(),
            kind.IsString() ? kind.Get<std::string>().c_str() : "unnamed");
    }
}

// One primitive as MikkTSpace sees it.
struct MikkPrimitive
{
    const glm::vec3* positions;
    const glm::vec3* normals;
    const glm::vec2* uvs;
    const std::vector<unsigned int>& indices;  // three per triangle
    std::vector<glm::vec4> corners;            // tangent and sign per corner

    unsigned int vertex(int face, int vert) const { return indices[face * 3 + vert]; }
};

const MikkPrimitive& mikkData(const SMikkTSpaceContext* context)
{
    return *static_cast<const MikkPrimitive*>(context->m_pUserData);
}

int mikkNumFaces(const SMikkTSpaceContext* context)
{
    return (int)(mikkData(context).indices.size() / 3);
}

int mikkNumVerticesOfFace(const SMikkTSpaceContext*, const int)
{
    return 3;
}

void mikkPosition(const SMikkTSpaceContext* context, float out[], const int face, const int vert)
{
    const MikkPrimitive& p = mikkData(context);
    const glm::vec3 v = p.positions[p.vertex(face, vert)];
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

void mikkNormal(const SMikkTSpaceContext* context, float out[], const int face, const int vert)
{
    const MikkPrimitive& p = mikkData(context);
    const glm::vec3 n = p.normals[p.vertex(face, vert)];
    out[0] = n.x;
    out[1] = n.y;
    out[2] = n.z;
}

// glTF's v runs down the image and MikkTSpace's up, so it gets 1 - v; only the bitangent sign changes.
void mikkTexCoord(const SMikkTSpaceContext* context, float out[], const int face, const int vert)
{
    const MikkPrimitive& p = mikkData(context);
    const glm::vec2 uv = p.uvs[p.vertex(face, vert)];
    out[0] = uv.x;
    out[1] = 1.0f - uv.y;
}

void mikkSetTangent(const SMikkTSpaceContext* context, const float tangent[], const float sign, const int face,
    const int vert)
{
    MikkPrimitive& p = *static_cast<MikkPrimitive*>(context->m_pUserData);
    p.corners[face * 3 + vert] = glm::vec4(tangent[0], tangent[1], tangent[2], sign < 0.0f ? -1.0f : 1.0f);
}

// State for one file; the maps keep entries shared by several nodes as one.
struct Loader
{
    const tinygltf::Model& model;
    const std::string& path;
    Scene& scene;
    int materialOverride;
    std::map<int, int> materialIds;                    // glTF material index (-1 = none) -> Scene material
    std::map<std::pair<int, int>, int> meshIds;        // (glTF mesh, primitive) -> Scene mesh, -1 if unusable
    std::map<std::pair<int, bool>, int> textureIds;    // (glTF texture, sRGB) -> Scene texture, -1 if unusable
    std::map<int, int> imageIds;                       // glTF image -> Scene texture image, -1 if unusable
    std::optional<GltfCamera> camera;

    void warn(const char* what, int meshIndex, int primIndex)
    {
        fprintf(stderr, "glTF %s: mesh %d primitive %d skipped: %s\n", path.c_str(), meshIndex, primIndex, what);
    }

    // Of the extensions' textures only transmissionTexture is read; Material's default is glTF's default material.
    int materialFor(int gltfMaterial)
    {
        if (materialOverride >= 0)
        {
            return materialOverride;
        }
        auto found = materialIds.find(gltfMaterial);
        if (found != materialIds.end())
        {
            return found->second;
        }
        Material m{};
        if (gltfMaterial >= 0 && gltfMaterial < (int)model.materials.size())
        {
            const tinygltf::Material& source = model.materials[gltfMaterial];
            const tinygltf::PbrMetallicRoughness& pbr = source.pbrMetallicRoughness;
            if (pbr.baseColorFactor.size() == 4)
            {
                m.baseColor = glm::vec3((float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1], (float)pbr.baseColorFactor[2]);
                m.alpha = (float)pbr.baseColorFactor[3];
            }
            m.metallic = (float)pbr.metallicFactor;
            m.roughness = (float)pbr.roughnessFactor;
            m.baseColorTexture = textureFor(pbr.baseColorTexture.index, pbr.baseColorTexture.texCoord, true);
            if (m.baseColorTexture < 0 && pbr.baseColorTexture.index >= 0)
            {
                // Magenta, so a broken asset shows in the render.
                m.baseColorTexture = missingTexture();
            }
            m.metallicRoughnessTexture = textureFor(pbr.metallicRoughnessTexture.index, pbr.metallicRoughnessTexture.texCoord, false);
            m.normalTexture = textureFor(source.normalTexture.index, source.normalTexture.texCoord, false);
            m.normalScale = (float)source.normalTexture.scale;

            const std::vector<double>& e = source.emissiveFactor;
            const glm::vec3 emissive = e.size() == 3 ? glm::vec3((float)e[0], (float)e[1], (float)e[2]) : glm::vec3(0.0f);
            m.emission = emissive * extensionNumber(source, "KHR_materials_emissive_strength", "emissiveStrength", 1.0f);
            if (maxComponent(m.emission) > 0.0f)
            {
                m.emissiveTexture = textureFor(source.emissiveTexture.index, source.emissiveTexture.texCoord, true);
            }

            if (source.alphaMode == "MASK")
            {
                m.alphaMode = ALPHA_MASK;
                m.alphaCutoff = (float)source.alphaCutoff;
            }
            else if (source.alphaMode == "BLEND")
            {
                m.alphaMode = ALPHA_BLEND;
            }

            // ior 0 means a Fresnel term of 1 in glTF; other values below 1 are invalid.
            m.ior = extensionNumber(source, "KHR_materials_ior", "ior", 1.5f);
            if (m.ior != 0.0f && m.ior < 1.0f)
            {
                m.ior = 1.0f;
            }

            m.transmission = glm::clamp(extensionNumber(source, "KHR_materials_transmission", "transmissionFactor", 0.0f), 0.0f, 1.0f);
            if (m.transmission > 0.0f)
            {
                const tinygltf::TextureInfo slot = extensionTexture(source, "KHR_materials_transmission", "transmissionTexture");
                m.transmissionTexture = textureFor(slot.index, slot.texCoord, false);
            }

            // Only the volume's absorption is read: every transmissive surface is solid, thickness is ignored.
            const float distance = extensionNumber(source, "KHR_materials_volume", "attenuationDistance", 0.0f);
            if (distance > 0.0f)
            {
                const glm::vec3 color = extensionColor(source, "KHR_materials_volume", "attenuationColor", glm::vec3(1.0f));
                m.absorption = -glm::log(glm::clamp(color, glm::vec3(1e-6f), glm::vec3(1.0f))) / distance;
            }

            m.specularFactor = glm::clamp(extensionNumber(source, "KHR_materials_specular", "specularFactor", 1.0f), 0.0f, 1.0f);
            m.specularColorFactor = glm::max(extensionColor(source, "KHR_materials_specular", "specularColorFactor", glm::vec3(1.0f)),
                glm::vec3(0.0f));
            m.clearcoat = glm::clamp(extensionNumber(source, "KHR_materials_clearcoat", "clearcoatFactor", 0.0f), 0.0f, 1.0f);
            m.clearcoatRoughness = glm::clamp(
                extensionNumber(source, "KHR_materials_clearcoat", "clearcoatRoughnessFactor", 0.0f), 0.0f, 1.0f);
        }
        const int id = (int)scene.materials.size();
        scene.materials.push_back(m);
        materialIds[gltfMaterial] = id;
        return id;
    }

    // The Scene texture for a slot, or -1 when it is empty or unusable. Only TEXCOORD_0 is loaded.
    int textureFor(int index, int texCoord, bool srgb)
    {
        if (index < 0)
        {
            return -1;
        }
        const std::pair<int, bool> key(index, srgb);
        auto found = textureIds.find(key);
        if (found != textureIds.end())
        {
            return found->second;
        }
        if (texCoord != 0)
        {
            fprintf(stderr, "glTF %s: texture %d reads TEXCOORD_%d, only TEXCOORD_0 is loaded\n",
                path.c_str(), index, texCoord);
        }
        int id = -1;
        const tinygltf::Texture* source = index < (int)model.textures.size() ? &model.textures[index] : nullptr;
        const int image = imageFor(source != nullptr ? source->source : -1);
        if (image >= 0)
        {
            Texture t;
            t.image = image;
            t.wrapU = cudaAddressModeWrap;
            t.wrapV = cudaAddressModeWrap;
            t.filter = cudaFilterModeLinear;
            t.srgb = srgb;
            if (source->sampler >= 0 && source->sampler < (int)model.samplers.size())
            {
                const tinygltf::Sampler& sampler = model.samplers[source->sampler];
                t.wrapU = addressMode(sampler.wrapS);
                t.wrapV = addressMode(sampler.wrapT);
                // No mipmaps: pixel jitter averages the texels under a pixel, so only magFilter applies.
                t.filter = sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
            }
            id = (int)scene.textures.size();
            scene.textures.push_back(t);
        }
        textureIds[key] = id;
        return id;
    }

    // One opaque magenta texel, so an alpha-tested surface stays visible.
    int missingTexture()
    {
        if (scene.missingTexture < 0)
        {
            TextureImage image;
            image.width = 1;
            image.height = 1;
            image.rgba = { 255, 0, 255, 255 };
            Texture t;
            t.image = (int)scene.textureImages.size();
            t.wrapU = cudaAddressModeWrap;
            t.wrapV = cudaAddressModeWrap;
            t.filter = cudaFilterModePoint;
            t.srgb = true;
            scene.textureImages.push_back(std::move(image));
            scene.missingTexture = (int)scene.textures.size();
            scene.textures.push_back(t);
        }
        return scene.missingTexture;
    }

    // -1 when tinygltf has no pixels (missing file, KTX2 or WebP via extension); 16-bit images round to 8 bits.
    int imageFor(int gltfImage)
    {
        auto found = imageIds.find(gltfImage);
        if (found != imageIds.end())
        {
            return found->second;
        }
        int id = -1;
        const tinygltf::Image* source =
            gltfImage >= 0 && gltfImage < (int)model.images.size() ? &model.images[gltfImage] : nullptr;
        const bool usable = source != nullptr && source->width > 0 && source->height > 0 && source->component == 4
            && (source->bits == 8 || source->bits == 16)
            && source->image.size() == (size_t)source->width * source->height * 4 * (source->bits / 8);
        if (!usable)
        {
            fprintf(stderr,
                "glTF %s: image %d has no pixels: a base color slot that uses it shows magenta, other slots use their factor\n",
                path.c_str(), gltfImage);
        }
        else
        {
            TextureImage image;
            image.width = source->width;
            image.height = source->height;
            if (source->bits == 8)
            {
                image.rgba = source->image;
            }
            else
            {
                image.rgba.resize((size_t)image.width * image.height * 4);
                const uint16_t* wide = reinterpret_cast<const uint16_t*>(source->image.data());
                for (size_t i = 0; i < image.rgba.size(); ++i)
                {
                    image.rgba[i] = (unsigned char)((wide[i] + 128) / 257);
                }
            }
            id = (int)scene.textureImages.size();
            scene.textureImages.push_back(std::move(image));
        }
        imageIds[gltfImage] = id;
        return id;
    }

    void cameraFor(int gltfCamera, const glm::mat4& world)
    {
        if (camera || gltfCamera < 0 || gltfCamera >= (int)model.cameras.size())
        {
            return;
        }
        const tinygltf::Camera& source = model.cameras[gltfCamera];
        // Negative when the transform mirrors, zero when a scale of 0 leaves no direction.
        const float det = glm::determinant(glm::mat3(world));
        if (source.type != "perspective" || source.perspective.yfov <= 0.0 || det == 0.0f)
        {
            fprintf(stderr, "glTF %s: camera %d skipped: not a usable perspective camera\n", path.c_str(), gltfCamera);
            return;
        }
        GltfCamera c;
        c.position = glm::vec3(world[3]);
        c.view = -glm::normalize(glm::vec3(world[2]));
        c.up = glm::normalize(glm::vec3(world[1]));
        c.mirrored = det < 0.0f;
        c.yfov = (float)source.perspective.yfov;
        c.aspectRatio = (float)source.perspective.aspectRatio;
        camera = c;
    }

    // Spot and directional lights shine down the node's -Z; dividing by 683 lm/W undoes Blender's candela and lux.
    void lightFor(int gltfLight, const glm::mat4& world, int group)
    {
        if (gltfLight < 0 || gltfLight >= (int)model.lights.size())
        {
            return;
        }
        const tinygltf::Light& source = model.lights[gltfLight];
        const glm::vec3 color = source.color.size() == 3
            ? glm::vec3((float)source.color[0], (float)source.color[1], (float)source.color[2]) : glm::vec3(1.0f);
        PunctualLight light{};
        light.intensity = color * (float)(source.intensity / LUMENS_PER_WATT);
        light.group = group;
        if (source.type == "point")
        {
            light.type = LIGHT_POINT;
            light.position = glm::vec3(world[3]);
        }
        else if (source.type == "spot")
        {
            light.type = LIGHT_SPOT;
            light.position = glm::vec3(world[3]);
            light.direction = -glm::normalize(glm::vec3(world[2]));
            const float outer = glm::clamp((float)source.spot.outerConeAngle, 0.0f, 0.5f * PI);
            const float inner = glm::clamp((float)source.spot.innerConeAngle, 0.0f, outer);
            light.cosOuter = cosf(outer);
            light.cosInner = cosf(inner);
        }
        else if (source.type == "directional")
        {
            light.type = LIGHT_DISTANT;
            light.position = glm::normalize(glm::vec3(world[2]));  // toward the light
        }
        else
        {
            fprintf(stderr, "glTF %s: light of unknown type \"%s\" ignored\n", path.c_str(), source.type.c_str());
            return;
        }
        if (maxComponent(light.intensity) > 0.0f)
        {
            scene.punctualLights.push_back(light);
        }
    }

    // The primitive's TriangleMesh index, or -1 if it cannot be used.
    int meshFor(int meshIndex, int primIndex)
    {
        const std::pair<int, int> key(meshIndex, primIndex);
        auto found = meshIds.find(key);
        if (found != meshIds.end())
        {
            return found->second;
        }
        const int id = loadPrimitive(meshIndex, primIndex);
        meshIds[key] = id;
        return id;
    }

    int loadPrimitive(int meshIndex, int primIndex)
    {
        const tinygltf::Primitive& prim = model.meshes[meshIndex].primitives[primIndex];
        if (prim.mode != TINYGLTF_MODE_TRIANGLES)
        {
            warn("not a triangle list (strips and fans are not supported)", meshIndex, primIndex);
            return -1;
        }
        std::string err;

        auto posAttr = prim.attributes.find("POSITION");
        AccessorView pos;
        if (posAttr == prim.attributes.end() || !viewAccessor(model, posAttr->second, pos, err) || !isFloatVec3(pos))
        {
            warn(err.empty() ? "no float POSITION attribute" : err.c_str(), meshIndex, primIndex);
            return -1;
        }
        const int baseVertex = (int)scene.positions.size();
        for (size_t i = 0; i < pos.count; ++i)
        {
            scene.positions.push_back(readVec3(pos, i));
        }

        // Stored offset by baseVertex, so they address the scene's flat arrays.
        std::vector<unsigned int> local;
        if (prim.indices >= 0)
        {
            AccessorView idx;
            if (!viewAccessor(model, prim.indices, idx, err) || idx.numComponents != 1 || !isIndexType(idx.componentType))
            {
                warn(err.empty() ? "unsupported index accessor" : err.c_str(), meshIndex, primIndex);
                scene.positions.resize(baseVertex);
                return -1;
            }
            local.resize(idx.count);
            for (size_t i = 0; i < idx.count; ++i)
            {
                local[i] = readIndex(idx, i);
            }
        }
        else
        {
            local.resize(pos.count);
            for (size_t i = 0; i < pos.count; ++i)
            {
                local[i] = (unsigned int)i;
            }
        }
        if (local.size() < 3 || local.size() % 3 != 0)
        {
            warn("index count is not a positive multiple of three", meshIndex, primIndex);
            scene.positions.resize(baseVertex);
            return -1;
        }
        const int triCount = (int)(local.size() / 3);
        const int indexOffset = (int)scene.indices.size();
        for (int t = 0; t < triCount; ++t)
        {
            const unsigned int a = local[3 * t];
            const unsigned int b = local[3 * t + 1];
            const unsigned int c = local[3 * t + 2];
            if (a >= pos.count || b >= pos.count || c >= pos.count)
            {
                warn("index out of range", meshIndex, primIndex);
                scene.positions.resize(baseVertex);
                scene.indices.resize(indexOffset);
                return -1;
            }
            scene.indices.push_back(glm::ivec3(baseVertex + a, baseVertex + b, baseVertex + c));
        }

        // Missing normals: area-weighted face normals (raw cross products) summed per vertex.
        auto nrmAttr = prim.attributes.find("NORMAL");
        AccessorView nrm;
        if (nrmAttr != prim.attributes.end() && viewAccessor(model, nrmAttr->second, nrm, err)
            && isFloatVec3(nrm) && nrm.count == pos.count)
        {
            for (size_t i = 0; i < nrm.count; ++i)
            {
                scene.normals.push_back(readVec3(nrm, i));
            }
        }
        else
        {
            std::vector<glm::vec3> sum(pos.count, glm::vec3(0.0f));
            for (int t = 0; t < triCount; ++t)
            {
                const glm::ivec3 tri = scene.indices[indexOffset + t] - glm::ivec3(baseVertex);
                const glm::vec3 p0 = scene.positions[baseVertex + tri.x];
                const glm::vec3 p1 = scene.positions[baseVertex + tri.y];
                const glm::vec3 p2 = scene.positions[baseVertex + tri.z];
                const glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
                sum[tri.x] += n;
                sum[tri.y] += n;
                sum[tri.z] += n;
            }
            for (size_t i = 0; i < pos.count; ++i)
            {
                scene.normals.push_back(glm::dot(sum[i], sum[i]) > 0.0f ? glm::normalize(sum[i]) : glm::vec3(0.0f, 0.0f, 1.0f));
            }
        }

        auto uvAttr = prim.attributes.find("TEXCOORD_0");
        AccessorView uv;
        std::string uvErr;
        bool hasUv = false;
        if (uvAttr != prim.attributes.end() && viewAccessor(model, uvAttr->second, uv, uvErr)
            && uv.numComponents == 2 && isUvType(uv.componentType) && uv.count == pos.count)
        {
            hasUv = true;
            for (size_t i = 0; i < uv.count; ++i)
            {
                scene.uvs.push_back(readUv(uv, i));
            }
        }
        else
        {
            if (uvAttr != prim.attributes.end())
            {
                fprintf(stderr, "glTF %s: mesh %d primitive %d: TEXCOORD_0 unusable (%s), uvs set to 0\n", path.c_str(),
                    meshIndex, primIndex, uvErr.empty() ? "unsupported layout" : uvErr.c_str());
            }
            scene.uvs.resize(scene.uvs.size() + pos.count, glm::vec2(0.0f));
        }

        // A zero tangent makes the shade kernel skip the normal map.
        auto tanAttr = prim.attributes.find("TANGENT");
        AccessorView tan;
        std::string tanErr;
        if (tanAttr != prim.attributes.end() && viewAccessor(model, tanAttr->second, tan, tanErr)
            && tan.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && tan.numComponents == 4 && tan.count == pos.count)
        {
            for (size_t i = 0; i < tan.count; ++i)
            {
                const float* f = reinterpret_cast<const float*>(tan.at(i));
                scene.tangents.push_back(glm::vec4(f[0], f[1], f[2], f[3] < 0.0f ? -1.0f : 1.0f));
            }
        }
        else
        {
            if (tanAttr != prim.attributes.end())
            {
                fprintf(stderr, "glTF %s: mesh %d primitive %d: TANGENT unusable (%s), generating it\n", path.c_str(),
                    meshIndex, primIndex, tanErr.empty() ? "unsupported layout" : tanErr.c_str());
            }
            const bool normalMapped = hasUv && materialOverride < 0 && prim.material >= 0 && prim.material < (int)model.materials.size()
                && model.materials[prim.material].normalTexture.index >= 0;
            if (normalMapped)
            {
                generateTangents(baseVertex, pos.count, indexOffset, triCount);
            }
            else
            {
                scene.tangents.resize(scene.tangents.size() + pos.count, glm::vec4(0.0f));
            }
        }

        TriangleMesh mesh;
        mesh.indexOffset = indexOffset;
        mesh.triCount = triCount;
        scene.meshes.push_back(mesh);
        return (int)scene.meshes.size() - 1;
    }

    // MikkTSpace tangents, as the glTF spec asks; a vertex whose corners disagree is split, one copy per tangent.
    void generateTangents(int baseVertex, size_t vertexCount, int indexOffset, int triCount)
    {
        TimingScope timing("load.mikktspace");
        std::vector<unsigned int> local((size_t)triCount * 3);
        for (int t = 0; t < triCount; ++t)
        {
            for (int k = 0; k < 3; ++k)
            {
                local[3 * t + k] = (unsigned int)(scene.indices[indexOffset + t][k] - baseVertex);
            }
        }
        MikkPrimitive data{ scene.positions.data() + baseVertex, scene.normals.data() + baseVertex,
            scene.uvs.data() + baseVertex, local, std::vector<glm::vec4>(local.size(), glm::vec4(0.0f)) };
        SMikkTSpaceInterface callbacks = {};
        callbacks.m_getNumFaces = mikkNumFaces;
        callbacks.m_getNumVerticesOfFace = mikkNumVerticesOfFace;
        callbacks.m_getPosition = mikkPosition;
        callbacks.m_getNormal = mikkNormal;
        callbacks.m_getTexCoord = mikkTexCoord;
        callbacks.m_setTSpaceBasic = mikkSetTangent;
        SMikkTSpaceContext context = {};
        context.m_pInterface = &callbacks;
        context.m_pUserData = &data;
        if (!genTangSpaceDefault(&context))
        {
            fprintf(stderr, "glTF %s: MikkTSpace failed, normal map skipped for a primitive\n", path.c_str());
            scene.tangents.resize(scene.tangents.size() + vertexCount, glm::vec4(0.0f));
            return;
        }

        struct Variant
        {
            glm::vec4 tangent;
            unsigned int vertex;
            int next;  // the vertex's next variant, -1 at the end
        };
        std::vector<int> firstVariant(vertexCount, -1);
        std::vector<Variant> variants;
        std::vector<glm::vec4> tangents(vertexCount, glm::vec4(0.0f));
        for (size_t c = 0; c < local.size(); ++c)
        {
            const unsigned int v = local[c];
            const glm::vec4 tangent = data.corners[c];
            int k = firstVariant[v];
            while (k >= 0 && variants[k].tangent != tangent)
            {
                k = variants[k].next;
            }
            unsigned int carrier;
            if (k >= 0)
            {
                carrier = variants[k].vertex;
            }
            else
            {
                carrier = v;
                if (firstVariant[v] >= 0)
                {
                    // Copied through locals: push_back may reallocate.
                    const glm::vec3 position = scene.positions[baseVertex + v];
                    const glm::vec3 normal = scene.normals[baseVertex + v];
                    const glm::vec2 uv = scene.uvs[baseVertex + v];
                    scene.positions.push_back(position);
                    scene.normals.push_back(normal);
                    scene.uvs.push_back(uv);
                    carrier = (unsigned int)tangents.size();
                    tangents.push_back(glm::vec4(0.0f));
                }
                tangents[carrier] = tangent;
                variants.push_back({ tangent, carrier, firstVariant[v] });
                firstVariant[v] = (int)variants.size() - 1;
            }
            scene.indices[indexOffset + c / 3][(int)(c % 3)] = baseVertex + (int)carrier;
        }
        scene.tangents.insert(scene.tangents.end(), tangents.begin(), tangents.end());
    }

    // Walks the node tree and makes one Geom per primitive.
    void visit(int nodeIndex, const glm::mat4& parent, int depth)
    {
        if (nodeIndex < 0 || nodeIndex >= (int)model.nodes.size() || depth > 64)
        {
            return;  // a malformed file: bad reference or a cycle
        }
        const tinygltf::Node& node = model.nodes[nodeIndex];
        const glm::mat4 world = parent * nodeLocalMatrix(node);
        const LightLink link = lightLinkFor(node, path);
        cameraFor(node.camera, world);
        lightFor(node.light, world, link.group);
        if (node.mesh >= 0 && node.mesh < (int)model.meshes.size())
        {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];
            for (size_t p = 0; p < mesh.primitives.size(); ++p)
            {
                const int meshId = meshFor(node.mesh, (int)p);
                if (meshId < 0)
                {
                    continue;
                }
                Geom g{};
                g.type = GeomType::MESH;
                g.materialId = materialFor(mesh.primitives[p].material);
                g.meshId = meshId;
                g.transform = world;
                g.inverseTransform = glm::inverse(world);
                g.invTranspose = glm::inverseTranspose(world);
                g.tangentSign = glm::determinant(glm::mat3(world)) < 0.0f ? -1.0f : 1.0f;
                g.lightMask = link.mask;
                g.lightGroup = link.group;
                scene.geoms.push_back(g);
            }
        }
        for (int child : node.children)
        {
            visit(child, world, depth + 1);
        }
    }
};
}  // namespace

// tinygltf decodes images during the file read, so this wrapper is the only way to time decoding apart.
static bool timedLoadImageData(tinygltf::Image* image, const int imageIndex, std::string* err, std::string* warn,
    int reqWidth, int reqHeight, const unsigned char* bytes, int size, void* userData)
{
    TimingScope timing("load.image_decode");
    return tinygltf::LoadImageData(image, imageIndex, err, warn, reqWidth, reqHeight, bytes, size, userData);
}

bool loadGltf(const std::string& path, const glm::mat4& sceneTransform, int materialOverride, Scene& scene,
    GltfInfo* info)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(timedLoadImageData, nullptr);
    std::string err;
    std::string warn;
    const bool binary = lowercaseExtension(path) == ".glb";
    bool loaded;
    {
        // load.gltf_parse is the read minus the decoding counted during it.
        const double decodeBefore = timingGet("load.image_decode");
        const auto start = std::chrono::steady_clock::now();
        loaded = binary ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                        : loader.LoadASCIIFromFile(&model, &err, &warn, path);
        const double readMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        timingAdd("load.gltf_parse", readMs - (timingGet("load.image_decode") - decodeBefore));
    }
    if (!warn.empty())
    {
        fprintf(stderr, "glTF %s: %s", path.c_str(), warn.c_str());
    }
    if (!loaded)
    {
        fprintf(stderr, "glTF %s: %s\n", path.c_str(), err.empty() ? "failed to load" : err.c_str());
        return false;
    }

    Loader l{ model, path, scene, materialOverride };
    const size_t firstGeom = scene.geoms.size();
    const size_t firstMesh = scene.meshes.size();
    const size_t firstTriangle = scene.indices.size();

    // A file with no scenes treats every node nobody names as a child as a root.
    {
        TimingScope timing("load.gltf_nodes");
        const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        if (sceneIndex < (int)model.scenes.size())
        {
            for (int root : model.scenes[sceneIndex].nodes)
            {
                l.visit(root, sceneTransform, 0);
            }
        }
        else
        {
            std::vector<bool> isChild(model.nodes.size(), false);
            for (const tinygltf::Node& node : model.nodes)
            {
                for (int child : node.children)
                {
                    if (child >= 0 && child < (int)isChild.size())
                    {
                        isChild[child] = true;
                    }
                }
            }
            for (size_t i = 0; i < model.nodes.size(); ++i)
            {
                if (!isChild[i])
                {
                    l.visit((int)i, sceneTransform, 0);
                }
            }
        }
    }

    glm::vec3 lo, hi;
    scene.bounds(firstGeom, scene.geoms.size(), lo, hi);
    printf("glTF %s: %zu triangles in %zu primitives, %zu instances, bounds (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)\n",
        path.c_str(), scene.indices.size() - firstTriangle, scene.meshes.size() - firstMesh,
        scene.geoms.size() - firstGeom, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
    reportUnusedLights(model, path, info != nullptr);
    if (materialOverride < 0)
    {
        reportUnsupportedMaterialFeatures(model, path);
    }

    if (info != nullptr)
    {
        info->boundsMin = lo;
        info->boundsMax = hi;
        info->camera = l.camera;
        info->render = renderHints(model);
        info->environment = pbrtEnvironment(model, path);
        addPbrtDistantLights(model, scene);
    }
    return true;
}
