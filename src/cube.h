#pragma once

// Built-in unit cube, ported from Cube.ts.
// Spans -0.5..0.5 on each axis, matching the base code's untransformed cube, so the
// JSON object's TRANS/ROTAT/SCALE becomes the TLAS instance transform exactly as before.
//
// 24 vertices rather than 8: each corner appears once per adjoining face, because the
// three faces meeting there have different normals. Winding is CCW viewed from outside
// (verified per face against the outward normal), matching icosphere.h and glTF.

#include "gpu/shared.h"
#include "sceneStructs.h"

#include <array>
#include <cstdint>

inline MeshData generateCube()
{
    // Face order: front (+z), back (-z), top (+y), bottom (-y), right (+x), left (-x).
    // Within each face the four vertices run bottom-left, bottom-right, top-right,
    // top-left in that face's own frame, which is what makes one UV pattern work for all
    static const std::array<glm::vec3, 24> positions = { {
            // Front
            {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f}, { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f},
            // Back
            { 0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f},
            // Top
            {-0.5f,  0.5f,  0.5f}, { 0.5f,  0.5f,  0.5f}, { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
            // Bottom
            {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f,  0.5f}, {-0.5f, -0.5f,  0.5f},
            // Right
            { 0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f,  0.5f},
            // Left
            {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f, -0.5f},
        } };

    static const std::array<glm::vec3, 6> faceNormals = { {
        { 0.0f,  0.0f,  1.0f},
        { 0.0f,  0.0f, -1.0f},
        { 0.0f,  1.0f,  0.0f},
        { 0.0f, -1.0f,  0.0f},
        { 1.0f,  0.0f,  0.0f},
        {-1.0f,  0.0f,  0.0f},
    } };

    static const std::array<glm::vec2, 4> faceUVs = { {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
    } };

    MeshPrimitive primitive;

    primitive.vertices.reserve(positions.size());
    for (uint32_t i = 0; i < positions.size(); ++i)
    {
        gpu::Vertex vertex;
        vertex.position = positions.at(i);
        vertex.normal = faceNormals.at(i / 4);
        vertex.uv = faceUVs.at(i % 4);
        primitive.vertices.push_back(vertex);
    }

    // Two triangles per quad, from the face's four vertices in BL/BR/TR/TL order.
    primitive.indices.reserve(36);
    for (uint32_t face = 0; face < 6; ++face)
    {
        const uint32_t base = face * 4;
        primitive.indices.insert(primitive.indices.end(), {
            base + 0, base + 1, base + 2,
            base + 0, base + 2, base + 3,
            });
    }

    MeshData mesh;
    mesh.primitives.push_back(std::move(primitive));
    return mesh;
}