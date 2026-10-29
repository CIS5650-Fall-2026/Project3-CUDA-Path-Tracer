#pragma once

#include "scene/environment.h"
#include "scene/lens.h"
#include "scene/sceneStructs.h"
#include <string>
#include <vector>

// Command-line overrides of the scene file; 0 or empty keeps the file's value.
struct SceneOverrides
{
    int width = 0;
    int height = 0;
    int iterations = 0;
    int traceDepth = 0;
    // Lat-long .hdr or .exr, at strength 1 and no rotation.
    std::string environmentFile;
    bool environmentCompensation = true;
    bool environmentLight = true;
    // "pinhole", "thin", or a lens file.
    std::string lens;
    // Scene units.
    float apertureRadius = 0.0f;
    float focusDistance = 0.0f;
    float fovy = 0.0f;           // degrees
    float unitsPerMeter = 0.0f;
};

struct RenderSettings
{
    glm::ivec2 resolution;
    int iterations;
    int traceDepth;
    std::string imageName;  // base name of auto-saved images
};

struct CameraPose
{
    glm::vec3 eye;
    glm::vec3 lookAt;  // also the point the interactive camera orbits
    glm::vec3 up;
    float fovy;        // full vertical field of view, degrees
    bool mirrored;
    CameraType type = CAMERA_PINHOLE;
    std::string lensFile;
    float apertureRadius = 0.0f;    // scene units
    float focusDistance = 0.0f;     // scene units; 0 = the distance to lookAt
    float unitsPerMeter = 1.0f;
};

// An existing file, else scenes/<name>.json, else the name's scenes/catalog.json entry; paths from the repo root.
std::string findSceneFile(const std::string& argument);

void listScenes();

// available is false for a catalog entry whose glTF is not downloaded.
struct SceneName
{
    std::string name;
    bool available;
};
// The scene JSONs in scenes/, sorted, then the catalog's entries in file order.
std::vector<SceneName> sceneNames();

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, const SceneOverrides& ov);
    void loadFromGltf(const std::string& gltfName, const SceneOverrides& ov);
    void initRenderState(const RenderSettings& settings, const CameraPose& pose);
public:
    // Rerun when the environment changes, since its power changes every light's share.
    void buildLights();
    // A scene JSON or a .gltf / .glb that is the whole scene; throws std::runtime_error when it cannot load.
    Scene(std::string filename, const SceneOverrides& ov = SceneOverrides());

    // World-space bounds of geoms [first, last); lo > hi when the range is empty.
    void bounds(size_t first, size_t last, glm::vec3& lo, glm::vec3& hi) const;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    // Flattened glTF meshes; each TriangleMesh says where its triangles are.
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;
    std::vector<glm::ivec3> indices;
    std::vector<TriangleMesh> meshes;
    std::vector<TextureImage> textureImages;
    std::vector<Texture> textures;
    // One magenta texel for unusable base color images, -1 until the glTF loader first needs it.
    int missingTexture = -1;
    EnvironmentSource environmentSource;
    Environment environment;
    // No surfaces when the camera names no lens file.
    LensSystem lens;
    bool hasLens() const { return !lens.surfaces.empty(); }

    // Emissive spheres are left to BSDF sampling; cubes join as their 12 triangles.
    std::vector<PunctualLight> punctualLights;
    std::vector<LightTriangle> lightTriangles;
    // One row per distinct Geom::lightMask, all ones first; row-major, entry [m * count + i].
    std::vector<unsigned int> lightMasks;
    // Per mask: triangles, then punctual lights, then the environment; each row ends at 1.
    std::vector<float> lightCdf;
    std::vector<float> punctualPickPdf;
    // 0 when there is no environment to sample or environmentLight is off.
    std::vector<float> environmentPickPdf;
    bool environmentLight = true;
    // Per mask and material: NEE's area density, the same for every triangle of the material.
    std::vector<float> emitterAreaPdf;
    // Entries per lightCdf row.
    int numLights = 0;
    RenderState state;
};
