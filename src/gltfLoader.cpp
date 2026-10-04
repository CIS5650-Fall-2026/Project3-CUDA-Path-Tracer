 
#define TINYGLTF_IMPLEMENTATION

// Dont include tinygl built in i/o
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include "gltfLoader.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

static void appendMeshTriangles(
  const tinygltf::Model& model,
  int meshIndex,
  const glm::mat4& worldTransform,
  std::vector<Triangle>& triangles
) {
  const auto& mesh = model.meshes.at(meshIndex);

  for (const auto& primitive : mesh.primitives) {
    int positionAccessorId = primitive.attributes.at("POSITION");
    const auto& positions = model.accessors.at(positionAccessorId);

    std::cout << "Vertex count: " << positions.count << std::endl;

    //if (primitive.indices >= 0) {
    //  const auto& indices = model.accessors.at(primitive.indices);
    //  std::cout << "Index count: " << indices.count << std::endl;
    //}

    const auto& view = model.bufferViews.at(positions.bufferView);
    const auto& buffer = model.buffers.at(view.buffer);

    int stride = positions.ByteStride(view);
    size_t start = view.byteOffset + positions.byteOffset;

    std::vector<glm::vec3> vertices;
    vertices.reserve(positions.count); // Preallocate

    for (size_t i = 0; i < positions.count; i++) {
      float xyz[3];
      size_t offset = start + i * static_cast<size_t>(stride);

      std::memcpy(xyz, buffer.data.data() + offset, sizeof(xyz));
      vertices.emplace_back(xyz[0], xyz[1], xyz[2]);
    }

    std::cout << "Read positions: " << vertices.size() << std::endl;

    // Read indices into a vector
    std::vector<uint32_t> vertexIndices;
    if (primitive.indices >= 0) {
      const auto& accessor = model.accessors.at(primitive.indices);
      const auto& indexView = model.bufferViews.at(accessor.bufferView);
      const auto& indexBuffer = model.buffers.at(indexView.buffer);

      size_t indexStart = indexView.byteOffset + accessor.byteOffset;
      int indexStride = accessor.ByteStride(indexView);

      vertexIndices.reserve(accessor.count);

      for (size_t i = 0; i < accessor.count; i++) {
        const unsigned char* bytes = indexBuffer.data.data() + indexStart + (i * static_cast<size_t>(indexStride));

        uint32_t value;

        switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
          uint8_t index;
          std::memcpy(&index, bytes, sizeof(index));
          value = index;
          break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
          uint16_t index;
          std::memcpy(&index, bytes, sizeof(index));
          value = index;
          break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
          std::memcpy(&value, bytes, sizeof(value));
          break;
        }
        default: {
          throw std::runtime_error("Invalid index format");
        }
        }

        vertexIndices.push_back(value);
      }
    }
    else {
      // No index list, assume vertices are consective
      for (size_t i = 0; i < vertices.size(); i++) {
        vertexIndices.push_back(static_cast<uint32_t>(i));
      }
    }

    std::cout << "Read indices: " << vertexIndices.size() << std::endl;

    size_t initialCount = triangles.size();
    for (size_t i = 0; i < vertexIndices.size(); i += 3) {
      Triangle tri;

      // Apply transform to vertices directly
      tri.v0 = glm::vec3(
        worldTransform * glm::vec4(
          vertices.at(vertexIndices.at(i)), 1.0f
        )
      );

      tri.v1 = glm::vec3(
        worldTransform * glm::vec4(
          vertices.at(vertexIndices.at(i + 1)), 1.0f
        )
      );

      tri.v2 = glm::vec3(
        worldTransform * glm::vec4(
          vertices.at(vertexIndices.at(i + 2)), 1.0f
        )
      );

      triangles.push_back(tri);
    }

    std::cout << "Added triangles: " << triangles.size() - initialCount << std::endl;
  }
}

static glm::mat4 getNodeTransform(const tinygltf::Node& node) {

  // If its 16, we have our full matrix given to us -> no need to compute it.
  if (node.matrix.size() == 16) {
    glm::mat4 result(1.0f);

    for (int col = 0; col < 4; col++) {
      for (int row = 0; row < 4; row++) {
        // DONT FORGET: glm matrices are column first
        result[col][row] = static_cast<float>(node.matrix[col * 4 + row]);
      }
    }

    return result;
  }
  
  // Have to compute our full transform matrix...
  glm::vec3 translation(0.0f);
  glm::vec3 scale(1.0f);
  glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

  if (node.translation.size() == 3) {
    translation = glm::vec3(
      node.translation[0],
      node.translation[1],
      node.translation[2]);
  }
    

  if (node.scale.size() == 3) {
    scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
  }
    

  if (node.rotation.size() == 4) {
    rotation = glm::quat(
      static_cast<float>(node.rotation[3]),
      static_cast<float>(node.rotation[0]),
      static_cast<float>(node.rotation[1]),
      static_cast<float>(node.rotation[2]));
  }
    

  return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

static void appendNodeTriangles(
  const tinygltf::Model& model,
  int nodeIndex,
  const glm::mat4& parentTransform,
  std::vector<Triangle>& triangles
) {
  const auto& node = model.nodes.at(nodeIndex);

  glm::mat4 Worldtransform = parentTransform * getNodeTransform(node);

  if (node.mesh >= 0)
    appendMeshTriangles(model, node.mesh, Worldtransform, triangles);

  for (int childIndex : node.children) {
    appendNodeTriangles(model, childIndex, Worldtransform, triangles);
  }
}

void loadGLTF(
  const std::string& filename,
  const glm::mat4& objectTransform,
  std::vector<Triangle>& triangles
) {
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;

  std::string errors;
  std::string warnings;

  /*
   * Loader parses the file
   * model reads its meshes, nodes, materials, etc (includes bianry buffers)
  */

  // Geometry only import
  loader.SetImageLoader(
    [](tinygltf::Image*, int, std::string*, std::string*,
      int, int, const unsigned char*, int, void*) {
        return true;
    },
    nullptr
  );

  const auto extension = std::filesystem::path(filename).extension().string();

  bool loaded;
  if (extension == ".glb") {
    loaded = loader.LoadBinaryFromFile(
      &model, &errors, &warnings, filename);
  }
  else if (extension == ".gltf") {
    loaded = loader.LoadASCIIFromFile(
      &model, &errors, &warnings, filename);
  }
  else {
    throw std::runtime_error("Expected .gltf or .glb");
  }

  if (!warnings.empty()) {
    std::cerr << warnings << std::endl;
  }


  if (!loaded) {
    throw std::runtime_error("Could not load " + filename + ": " + errors);
  }

  // Mesh has been successfully read, now load relevant information
  //appendMeshTriangles(model, 0, objectTransform, triangles);

  if (model.scenes.empty()) {
    std::cerr << "No gltf scene imported" << std::endl;
    return;
  }
  
  int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
  const auto& scene = model.scenes.at(sceneIndex);

  for (int rootNodeIndex : scene.nodes) {
    appendNodeTriangles(model, rootNodeIndex, objectTransform, triangles);
  }
}

/**
* "Test" function to check if gltf reading is working correctly. 
*/
void inspectGLTF(const std::string& filename)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;

  std::string errors;
  std::string warnings;
  
  /*
   * Loader parses the file
   * model reads its meshes, nodes, materials, etc (includes bianry buffers)
  */

  // Geometry only import
  loader.SetImageLoader(
      [](tinygltf::Image*, int, std::string*, std::string*,
      int, int, const unsigned char*, int, void*) {
        return true;
    },
    nullptr
  );

  const auto extension = std::filesystem::path(filename).extension().string();

  bool loaded;
  if (extension == ".glb") {
    loaded = loader.LoadBinaryFromFile(
      &model, &errors, &warnings, filename);
  }
  else if (extension == ".gltf") {
    loaded = loader.LoadASCIIFromFile(
      &model, &errors, &warnings, filename);
  }
  else {
    throw std::runtime_error("Expected .gltf or .glb");
  }

  if (!warnings.empty()) {
    std::cerr << warnings << std::endl;
  }
   

  if (!loaded) {
    throw std::runtime_error("Could not load " + filename + ": " + errors);
  }

  std::cout << "Meshes: "       << model.meshes.size()
            << "\nNodes: "      << model.nodes.size()
            << "\nMaterials: "  << model.materials.size()
            << std::endl;
}