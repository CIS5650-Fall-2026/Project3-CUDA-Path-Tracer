#include "scene.h"

#include "octree.h"
#include "utilities.h"

#include "json.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tiny_gltf.h"

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace {
bool gltfSkipImage(tinygltf::Image *, const int, std::string *, std::string *,
                   int, int, const unsigned char *, int, void *) {
  return true;
}

glm::mat4 gltfNodeMatrix(const tinygltf::Node &node) {
  if (node.matrix.size() == 16) {
    glm::mat4 m;
    for (int i = 0; i < 16; i++) {
      m[i / 4][i % 4] = (float)node.matrix[i];
    }
    return m;
  }

  glm::mat4 T, R, S;
  if (node.translation.size() == 3) {
    T = glm::translate(glm::mat4(),
                       glm::vec3(node.translation[0], node.translation[1],
                                 node.translation[2]));
  }
  if (node.rotation.size() == 4) {
    // glTF stores (x, y, z, w) -> glm::quat takes (w, x, y, z)
    R = glm::mat4_cast(
        glm::quat((float)node.rotation[3], (float)node.rotation[0],
                  (float)node.rotation[1], (float)node.rotation[2]));
  }
  if (node.scale.size() == 3) {
    S = glm::scale(glm::mat4(),
                   glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
  }
  return T * R * S;
}

// returns a pointer to element 0 of an accessor and its byte stride.
const unsigned char *gltfAccessorData(const tinygltf::Model &model,
                                      const tinygltf::Accessor &accessor,
                                      size_t &stride) {
  const tinygltf::BufferView &view = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer &buffer = model.buffers[view.buffer];
  int s = accessor.ByteStride(view);
  stride = s > 0 ? (size_t)s : 0;
  return buffer.data.data() + view.byteOffset + accessor.byteOffset;
}

std::vector<glm::vec3> gltfReadVec3(const tinygltf::Model &model,
                                    int accessorIndex) {
  const tinygltf::Accessor &accessor = model.accessors[accessorIndex];
  std::vector<glm::vec3> out(accessor.count);
  if (accessor.bufferView < 0 ||
      accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
      accessor.type != TINYGLTF_TYPE_VEC3) {
    return {};
  }
  size_t stride;
  const unsigned char *data = gltfAccessorData(model, accessor, stride);
  for (size_t i = 0; i < accessor.count; i++) {
    const float *f = reinterpret_cast<const float *>(data + i * stride);
    out[i] = glm::vec3(f[0], f[1], f[2]);
  }
  return out;
}

std::vector<uint32_t> gltfReadIndices(const tinygltf::Model &model,
                                      int accessorIndex) {
  const tinygltf::Accessor &accessor = model.accessors[accessorIndex];
  std::vector<uint32_t> out(accessor.count);
  size_t stride;
  const unsigned char *data = gltfAccessorData(model, accessor, stride);
  for (size_t i = 0; i < accessor.count; i++) {
    const unsigned char *p = data + i * stride;
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      out[i] = *p;
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      out[i] = *reinterpret_cast<const uint16_t *>(p);
      break;
    default:
      out[i] = *reinterpret_cast<const uint32_t *>(p);
      break;
    }
  }
  return out;
}

void gltfAppendMesh(const tinygltf::Model &model, const tinygltf::Mesh &mesh,
                    const glm::mat4 &xform, std::vector<Triangle> &out) {
  glm::mat3 normalXform = glm::mat3(glm::inverseTranspose(xform));

  for (const tinygltf::Primitive &prim : mesh.primitives) {
    if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1) {
      continue;
    }
    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end()) {
      continue;
    }

    std::vector<glm::vec3> positions = gltfReadVec3(model, posIt->second);
    std::vector<glm::vec3> normals;
    auto nrmIt = prim.attributes.find("NORMAL");
    if (nrmIt != prim.attributes.end()) {
      normals = gltfReadVec3(model, nrmIt->second);
    }
    bool hasNormals = normals.size() == positions.size();

    std::vector<uint32_t> indices;
    if (prim.indices >= 0) {
      indices = gltfReadIndices(model, prim.indices);
    } else {
      indices.resize(positions.size());
      for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = (uint32_t)i;
      }
    }

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
      uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
      if (i0 >= positions.size() || i1 >= positions.size() ||
          i2 >= positions.size()) {
        continue;
      }

      Triangle tri;
      tri.v0 = glm::vec3(xform * glm::vec4(positions[i0], 1.0f));
      tri.v1 = glm::vec3(xform * glm::vec4(positions[i1], 1.0f));
      tri.v2 = glm::vec3(xform * glm::vec4(positions[i2], 1.0f));

      if (hasNormals) {
        tri.n0 = glm::normalize(normalXform * normals[i0]);
        tri.n1 = glm::normalize(normalXform * normals[i1]);
        tri.n2 = glm::normalize(normalXform * normals[i2]);
      } else {
        glm::vec3 n =
            glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
        tri.n0 = tri.n1 = tri.n2 = n;
      }
      out.push_back(tri);
    }
  }
}

// an empty nodeName loads every node; otherwise only that node and its children
void gltfAppendNode(const tinygltf::Model &model, int nodeIndex,
                    const glm::mat4 &parent, const std::string &nodeName,
                    bool included, std::vector<Triangle> &out) {
  const tinygltf::Node &node = model.nodes[nodeIndex];
  glm::mat4 world = parent * gltfNodeMatrix(node);
  included = included || nodeName.empty() || node.name == nodeName;
  if (included && node.mesh >= 0) {
    gltfAppendMesh(model, model.meshes[node.mesh], world, out);
  }
  for (int child : node.children) {
    gltfAppendNode(model, child, world, nodeName, included, out);
  }
}
} // namespace

void Scene::loadGLTF(const std::string &path, bool normalize,
                     const std::string &nodeName, Geom &geom) {
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  loader.SetImageLoader(gltfSkipImage, nullptr);

  bool isBinary = path.size() >= 4 && path.substr(path.size() - 4) == ".glb";
  bool ok = isBinary ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                     : loader.LoadASCIIFromFile(&model, &err, &warn, path);

  if (!warn.empty()) {
    cout << "glTF warning: " << warn << endl;
  }
  if (!ok) {
    cout << "Failed to load glTF " << path << ": " << err << endl;
    exit(-1);
  }

  std::vector<Triangle> meshTris;
  if (!model.scenes.empty()) {
    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    for (int root : model.scenes[sceneIndex].nodes) {
      gltfAppendNode(model, root, glm::mat4(), nodeName, false, meshTris);
    }
  } else {
    for (const tinygltf::Mesh &mesh : model.meshes) {
      gltfAppendMesh(model, mesh, glm::mat4(), meshTris);
    }
  }

  if (meshTris.empty()) {
    cout << "glTF " << path << " has no triangles"
         << (nodeName.empty() ? "" : " under node " + nodeName) << endl;
  }

  // recenter and fit the model into a unit cube so TRANS/SCALE behave the same
  // for any model.
  glm::mat4 fit;
  if (normalize && !meshTris.empty()) {
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (const Triangle &t : meshTris) {
      lo = glm::min(lo, glm::min(t.v0, glm::min(t.v1, t.v2)));
      hi = glm::max(hi, glm::max(t.v0, glm::max(t.v1, t.v2)));
    }
    glm::vec3 extent = hi - lo;
    float maxExtent = std::max(extent.x, std::max(extent.y, extent.z));
    float s = maxExtent > 0.0f ? 1.0f / maxExtent : 1.0f;
    fit = glm::scale(glm::mat4(), glm::vec3(s)) *
          glm::translate(glm::mat4(), -0.5f * (lo + hi));
  }

  glm::mat4 xform = geom.transform * fit;
  glm::mat3 normalXform = glm::mat3(glm::inverseTranspose(xform));

  geom.triangleStart = (int)triangles.size();
  geom.triangleCount = (int)meshTris.size();
  geom.bboxMin = glm::vec3(FLT_MAX);
  geom.bboxMax = glm::vec3(-FLT_MAX);

  for (Triangle t : meshTris) {
    t.v0 = glm::vec3(xform * glm::vec4(t.v0, 1.0f));
    t.v1 = glm::vec3(xform * glm::vec4(t.v1, 1.0f));
    t.v2 = glm::vec3(xform * glm::vec4(t.v2, 1.0f));
    t.n0 = glm::normalize(normalXform * t.n0);
    t.n1 = glm::normalize(normalXform * t.n1);
    t.n2 = glm::normalize(normalXform * t.n2);

    geom.bboxMin = glm::min(geom.bboxMin, glm::min(t.v0, glm::min(t.v1, t.v2)));
    geom.bboxMax = glm::max(geom.bboxMax, glm::max(t.v0, glm::max(t.v1, t.v2)));
    triangles.push_back(t);
  }
  geom.bboxMin -= glm::vec3(1e-4f);
  geom.bboxMax += glm::vec3(1e-4f);

  cout << "Loaded " << path << ": " << geom.triangleCount << " triangles"
       << endl;

  geom.octreeRoot = buildMeshOctree(
      triangles, geom.triangleStart, geom.triangleCount, geom.bboxMin,
      geom.bboxMax, octreeNodes, octreeTriIndices);
}

Scene::Scene(string filename) {
  cout << "Reading scene from " << filename << " ..." << endl;
  cout << " " << endl;
  auto ext = filename.substr(filename.find_last_of('.'));
  if (ext == ".json") {
    loadFromJSON(filename);
    return;
  } else {
    cout << "Couldn't read from " << filename << endl;
    exit(-1);
  }
}

void Scene::loadFromJSON(const std::string &jsonName) {
  std::ifstream f(jsonName);
  json data = json::parse(f);
  const auto &materialsData = data["Materials"];
  std::unordered_map<std::string, uint32_t> MatNameToID;
  for (const auto &item : materialsData.items()) {
    const auto &name = item.key();
    const auto &p = item.value();
    Material newMaterial{};

    if (p["TYPE"] == "Diffuse") {
      const auto &col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
    } else if (p["TYPE"] == "Emitting") {
      const auto &col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
      newMaterial.emittance = p["EMITTANCE"];
    } else if (p["TYPE"] == "Specular") {
      const auto &col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
      newMaterial.specular.color = newMaterial.color;
      newMaterial.hasReflective = 1.0f;
      newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
    }
    // diffuse base with a mirror coat that is hit REFLECTIVITY of the time
    else if (p["TYPE"] == "Glossy") {
      const auto &col = p["RGB"];
      newMaterial.color = glm::vec3(col[0], col[1], col[2]);
      newMaterial.specular.color = glm::vec3(1.0f);
      if (p.contains("SPEC_RGB")) {
        const auto &spec = p["SPEC_RGB"];
        newMaterial.specular.color = glm::vec3(spec[0], spec[1], spec[2]);
      }
      newMaterial.hasReflective = p.value("REFLECTIVITY", 0.5f);
      newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
    }
    MatNameToID[name] = materials.size();
    materials.emplace_back(newMaterial);
  }
  size_t lastSlash = jsonName.find_last_of("/\\");
  std::string sceneDir =
      lastSlash == std::string::npos ? "" : jsonName.substr(0, lastSlash + 1);

  const auto &objectsData = data["Objects"];
  for (const auto &p : objectsData) {
    const auto &type = p["TYPE"];
    Geom newGeom{};
    newGeom.triangleStart = -1;
    newGeom.octreeRoot = -1;
    if (type == "cube") {
      newGeom.type = CUBE;
    } else if (type == "mesh") {
      newGeom.type = MESH;
    } else {
      newGeom.type = SPHERE;
    }
    newGeom.materialid = MatNameToID[p["MATERIAL"]];
    const auto &trans = p["TRANS"];
    const auto &rotat = p["ROTAT"];
    const auto &scale = p["SCALE"];
    newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
    newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
    newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
    newGeom.transform = utilityCore::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    if (newGeom.type == MESH) {
      bool normalize =
          p.contains("NORMALIZE") ? p["NORMALIZE"].get<bool>() : true;
      std::string nodeName = p.value("NODE", std::string());
      loadGLTF(sceneDir + p["FILE"].get<std::string>(), normalize, nodeName,
               newGeom);
    }

    geoms.push_back(newGeom);
  }
  const auto &cameraData = data["Camera"];
  Camera &camera = state.camera;
  RenderState &state = this->state;
  camera.resolution.x = cameraData["RES"][0];
  camera.resolution.y = cameraData["RES"][1];
  float fovy = cameraData["FOVY"];
  state.iterations = cameraData["ITERATIONS"];
  state.traceDepth = cameraData["DEPTH"];
  state.imageName = cameraData["FILE"];
  const auto &pos = cameraData["EYE"];
  const auto &lookat = cameraData["LOOKAT"];
  const auto &up = cameraData["UP"];
  camera.position = glm::vec3(pos[0], pos[1], pos[2]);
  camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
  camera.up = glm::vec3(up[0], up[1], up[2]);

  // calculate fov based on resolution
  float yscaled = tan(fovy * (PI / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / PI;
  camera.fov = glm::vec2(fovx, fovy);

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                 2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  // set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
