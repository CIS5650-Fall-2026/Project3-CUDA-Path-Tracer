#include "scene.h"

#include "utilities.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "stb_image.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cfloat>
#include <functional>

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

int Scene::loadTexture(const std::string& texName) {
    int w;
    int h;
    int channels;
	float* data = stbi_loadf(texName.c_str(), &w, &h, &channels, 4);
    if (!data) {
		std::cerr << "std image: Failed to load texture!" << texName << std::endl;
        exit(-1);
    }
    
    Texture tex{
		w, h, (int)texels.size()
    };

    // do 4 channels for rgba
    for (int i = 0; i < w * h; ++i) {
        texels.emplace_back(
            data[4 * i], 
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3]
        );
    }
    stbi_image_free(data);
	textures.push_back(tex);

    return (int)textures.size() - 1;
}

/**
* This is a recursive BVH builder.  Recursive bvh picks a median split point for the axis
* using a splitting heuristic.
* Tris can then be sorted along this axis.
* The resulting node is an interior node with left and right children equal to recursive calls
* over the ranges start -> middle and middle -> end.
*/
int Scene::buildBVH(int start, int end) {
    int nodeIdx = (int)bvhNodes.size();
    bvhNodes.emplace_back();
    glm::vec3 bMin(FLT_MAX);
    glm::vec3 bMax(-FLT_MAX);

    for (int i = start; i < end; ++i) {
        const Triangle& tri = triangles[i];
        bMin = glm::min(bMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
        bMax = glm::max(bMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
    }

    int left = -1;
    int right = -1;

    if (end - start > BVH_LEAF_SIZE) {
        glm::vec3 ext = bMax - bMin;
        int axis = 0;
        if (ext.y > ext[axis]) {
            axis = 1;
        }
        if (ext.z > ext[axis]) {
            axis = 2;
        }

        // median split needs triangles to be partitioned about the middle (not fully ordered)
        // this lets each level be in linear time
        // then, node reference will be taken after recursive calls have been made
        int mid = start + (end - start) / 2;
        std::nth_element(
            triangles.begin() + start,
            triangles.begin() + mid,
            triangles.begin() + end,
            [axis](const Triangle& a, const Triangle& b) {
            return a.v0[axis] + a.v1[axis] + a.v2[axis]
                < b.v0[axis] + b.v1[axis] + b.v2[axis];
            }
        );

        left = buildBVH(start, mid);
        right = buildBVH(mid, end);
    }

    BVHNode& node = bvhNodes[nodeIdx];
    node.bboxMin = bMin;
    node.bboxMax = bMax;
    node.left = left;
    node.right = right;
    node.triStart = start;
    node.numTris = (left < 0) ? end - start : 0;
    return nodeIdx;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    // we will retrieve from file path
    const std::string basePath = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);

    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.albedoTexId = -1;
        newMaterial.bumpTexId = -1;
        newMaterial.bumpStrength = 1.f;
        newMaterial.noiseId = 0;
        newMaterial.noiseScale = 1.f;
        newMaterial.warpStrength = 1.f;
        newMaterial.warpFreq = 1.f;
        newMaterial.sssDensity = 0.f;

        // handle materials loading differently
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
            newMaterial.hasReflective = 1.f;
        }
        else if (p["TYPE"] == "Transmissive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Glass") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 1.f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Subsurface") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.sssDensity = p["SSS_DENSITY"];
        }

        if (p.contains("TEXTURE")) {
            const std::string file = p["TEXTURE"];
            newMaterial.albedoTexId = loadTexture(basePath + file);
        }

        if (p.contains("BUMP")) {
            const std::string file = p["BUMP"];
            newMaterial.bumpTexId = loadTexture(basePath + file);
        }

        if (p.contains("BUMP_STRENGTH")) {
            newMaterial.bumpStrength = p["BUMP_STRENGTH"];
        }

        if (p.contains("NOISE")) {
            const std::string type = p["NOISE"];
            if (type == "fbm") {
                newMaterial.noiseId = 1;
            }
            else if (type == "voronoi") {
                newMaterial.noiseId = 2;
            }
            else if (type == "fbm voronoi") {
                newMaterial.noiseId = 3;
            }
        }

        if (p.contains("NOISE_SCALE")) {
            newMaterial.noiseScale = p["NOISE_SCALE"];
        }

        if (p.contains("WARP_STRENGTH")) {
            newMaterial.warpStrength = p["WARP_STRENGTH"];
        }

        if (p.contains("WARP_FREQUENCY")) {
            newMaterial.warpFreq = p["WARP_FREQUENCY"];
        }

        // to prevent colors from washing out, I'm converting them to linear
        newMaterial.color = glm::pow(newMaterial.color, glm::vec3(2.2f));

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    // hdri env map support
    if (data.contains("Environment")) {
        const auto& env = data["Environment"];
        const std::string file = env["FILE"];
        envTexId = loadTexture(basePath + file);
        
        if (env.contains("INTENSITY")) {
            envIntensity = env["INTENSITY"];
        }
    }
    
    static const std::unordered_map<std::string, GeomType> geomType = {
        {"sphere", SPHERE},
        {"cube", CUBE},
        {"mesh", MESH}
    };

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
		auto it = geomType.find(type);
        if (it == geomType.end()) {
			cout << "Unknown object type!" << type << endl;
            exit(-1);
        }

        Geom newGeom{};
		newGeom.type = it->second;

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

        // for cube case
        newGeom.area = 2.f * (newGeom.scale.x * newGeom.scale.y
            + newGeom.scale.y * newGeom.scale.z
            + newGeom.scale.x * newGeom.scale.z
            );

        if (type == "cube" && materials[newGeom.materialid].emittance > 0.f) {
            lightIds.push_back((int)geoms.size());
        }

        // load mesh only after matrices exist so we can transform everything
        if (type == "mesh")
        {
            newGeom.type = MESH;
            const std::string file = p["FILE"];
            loadMeshFromOBJ(basePath + file, newGeom);
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

    // for handling depth
    camera.lensRadius = cameraData.contains("LENS_RADIUS")
        ? cameraData["LENS_RADIUS"].get<float>()
        : 0.f;
    camera.focalDistance = cameraData.contains("FOCAL_DISTANCE")
        ? cameraData["FOCAL_DISTANCE"].get<float>()
        : glm::length(camera.lookAt - camera.position);

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

// OBJ triangulated mesh loading.  I am adapting from tinyOBJLoader:
// https://github.com/tinyobjloader/tinyobjloader
// This is cpu side loading, and later the GPU will read them via ray-triangle intersection testing.
void Scene::loadMeshFromOBJ(const std::string& objName, Geom& geom) {
    tinyobj::ObjReaderConfig config;
	config.triangulate = true;
	tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objName, config)) {
		std::cerr << "TinyObjReader loading failed: " << reader.Error() << std::endl;
		exit(-1);
    }
    if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader warning: " << reader.Warning() << std::endl;
    }

	const tinyobj::attrib_t& attrib = reader.GetAttrib();
	const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

	const std::vector<tinyobj::real_t>& positions = attrib.vertices;
	const std::vector<tinyobj::real_t>& normals = attrib.normals;
	const std::vector<tinyobj::real_t>& uvs = attrib.texcoords;

    auto position = [&](const tinyobj::index_t& i) {
        glm::vec3 p(
            positions[3 * i.vertex_index + 0],
            positions[3 * i.vertex_index + 1],
            positions[3 * i.vertex_index + 2]
        );
        return glm::vec3(geom.transform * glm::vec4(p, 1.f));
    };

    auto normal = [&](const tinyobj::index_t& i) {
        glm::vec3 n(
            normals[3 * i.normal_index + 0],
            normals[3 * i.normal_index + 1],
            normals[3 * i.normal_index + 2]
        );
		return glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(n, 0.f)));
	};

    // note that uv i is at 2i, 2i + 1 in the array
    auto uv = [&](const tinyobj::index_t& i) {
        glm::vec2 uv(
            uvs[2 * i.texcoord_index + 0],
            uvs[2 * i.texcoord_index + 1]
        );
        return uv;
	};

	geom.triStart = (int)triangles.size();
    geom.bboxMin = glm::vec3(FLT_MAX);
	geom.bboxMax = glm::vec3(-FLT_MAX);

    for (size_t i = 0; i < shapes.size(); ++i) {
		const std::vector<tinyobj::index_t>& indices = shapes[i].mesh.indices;

        for (size_t j = 0; j + 2 < indices.size(); j += 3) {
			const tinyobj::index_t& i1 = indices[j];
			const tinyobj::index_t& i2 = indices[j + 1];
			const tinyobj::index_t& i3 = indices[j + 2];

            Triangle tri{};
			tri.v0 = position(i1);
			tri.v1 = position(i2);
			tri.v2 = position(i3);

            if (i1.normal_index >= 0 && i2.normal_index >= 0 && i3.normal_index >= 0) {
                tri.n0 = normal(i1);
                tri.n1 = normal(i2);
                tri.n2 = normal(i3);
            }
            else {
				tri.n0 = tri.n1 = tri.n2 = 
                    glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
            }

            // set uvs
            if (i1.texcoord_index >= 0 && i2.texcoord_index >= 0 && i3.texcoord_index >= 0) {
				tri.uv0 = uv(i1);
				tri.uv1 = uv(i2);
				tri.uv2 = uv(i3);

                // compute tangent
				glm::vec3 e1 = tri.v1 - tri.v0;
                glm::vec3 e2 = tri.v2 - tri.v0;
				glm::vec2 d1 = tri.uv1 - tri.uv0;
				glm::vec2 d2 = tri.uv2 - tri.uv0;
				float determinant = d1.x * d2.y - d1.y * d2.x;
                if (fabs(determinant) > 1e-8f) {
                    tri.tangent = glm::normalize((e1 * d2.y - e2 * d1.y) / determinant);
                }
            }

			geom.bboxMin = glm::min(geom.bboxMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
			geom.bboxMax = glm::max(geom.bboxMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
			triangles.push_back(tri);
        }
    }

	geom.numTris = (int)triangles.size() - geom.triStart;
    geom.bvhRoot = buildBVH(geom.triStart, geom.triStart + geom.numTris);

    // show the tree depth for debugging purposes
    std::function<int(int)> depthOf = [&](int i) {
        const BVHNode& n = bvhNodes[i];
        return n.left < 0 ? 1 : 1 + std::max(depthOf(n.left), depthOf(n.right));
        };
    std::cout << geom.numTris << " triangles, BVH depth " << depthOf(geom.bvhRoot)
        << ", " << bvhNodes.size() << " nodes total" << std::endl;
}
