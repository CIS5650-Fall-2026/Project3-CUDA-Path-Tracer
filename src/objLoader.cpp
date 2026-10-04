#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "objLoader.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

void loadObj(
  const std::string& filename,
  const glm::mat4& objectTransform,
  std::vector<Triangle>& triangles) {

  tinyobj::ObjReaderConfig config;
  config.triangulate = true;

  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(filename, config)) {
    throw std::runtime_error(reader.Error());
  }

  // Prints mostly material related warnings/
  if (!reader.Warning().empty()) {
    std::cerr << reader.Warning() << std::endl;
  }

  // Start reading
  const auto& attributes = reader.GetAttrib();

  // lambda for simpler access
  auto readVertex = [&](const tinyobj::index_t& index) {
    size_t start = static_cast<size_t>(index.vertex_index) * 3;

    glm::vec3 position(
      attributes.vertices.at(start),
      attributes.vertices.at(start + 1),
      attributes.vertices.at(start + 2)
    );

    return glm::vec3(
      objectTransform * glm::vec4(position, 1.0f)
    );
    };

  size_t initialCount = triangles.size();

  for (const auto& shape : reader.GetShapes()) {
    size_t offset = 0;

    for (auto faceSize : shape.mesh.num_face_vertices) {
      if (faceSize != 3) {
        throw std::runtime_error("Non-triangular faces not supported");
      }

      Triangle triangle;
      triangle.v0 = readVertex(shape.mesh.indices.at(offset));
      triangle.v1 = readVertex(shape.mesh.indices.at(offset + 1));
      triangle.v2 = readVertex(shape.mesh.indices.at(offset + 2));

      triangles.push_back(triangle);
      offset += faceSize;
    }
  }

  std::cout << "Added OBJ triangles: " << triangles.size() - initialCount << std::endl;
}
