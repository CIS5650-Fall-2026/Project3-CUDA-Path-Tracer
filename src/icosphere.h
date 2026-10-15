#pragma once

// Icosphere generator, ported from Icosphere.ts.
// Produces a unit-diameter sphere (radius 0.5, centered at the origin) in object
// space, matching the base code's untransformed sphere. The JSON object's
// TRANS/ROTAT/SCALE becomes the TLAS instance transform, exactly like cubes.

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "gpu/shared.h"
#include "sceneStructs.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

// subdivisions: 0 = icosahedron (20 tris); each level multiplies triangles by 4.
// 4 → 5,120 tris / 2,562 verts; 5 → 20,480 tris / 10,242 verts.
inline MeshData generateIcosphere(int subdivisions, float radius = 0.5f)
{
    const float X = 0.525731112119133606f;
    const float Z = 0.850650808352039932f;
    const float N = 0.0f;

    // Unit-length directions; scaled to `radius` at the end.
    std::vector<glm::vec3> dirs = {
        {-X, N,  Z}, { X, N,  Z}, {-X, N, -Z}, { X, N, -Z},
        { N, Z,  X}, { N, Z, -X}, { N, -Z, X}, { N, -Z, -X},
        { Z, X,  N}, {-Z, X,  N}, { Z, -X, N}, {-Z, -X, N},
    };

    // The original TypeScript table winds every face clockwise when seen from
    // outside. Swapping the last two indices makes them CCW-outward, matching
    // glTF, so one front-face rule works for all meshes.
    std::vector<std::array<uint32_t, 3>> tris = {
        {0,1,4},  {0,4,9},  {9,4,5},  {4,8,5},  {4,1,8},
        {8,1,10}, {8,10,3}, {5,8,3},  {5,3,2},  {2,3,7},
        {7,3,10}, {7,10,6}, {7,6,11}, {11,6,0}, {0,6,1},
        {6,10,1}, {9,11,0}, {9,2,11}, {9,5,2},  {7,11,2},
    };

    const size_t finalTris  = 20ull << (2 * subdivisions);          // 20 * 4^s
    const size_t finalVerts = 10ull * (1ull << (2 * subdivisions)) + 2;
    dirs.reserve(finalVerts);

    for (int s = 0; s < subdivisions; ++s)
    {
        // Maps an undirected edge (packed as min<<32 | max) to its midpoint vertex.
        std::unordered_map<uint64_t, uint32_t> edgeMap;
        edgeMap.reserve(tris.size() * 3 / 2);

        auto mid = [&](uint32_t a, uint32_t b) -> uint32_t {
            const uint64_t key = (uint64_t(glm::min(a, b)) << 32) | glm::max(a, b);
            auto it = edgeMap.find(key);
            if (it != edgeMap.end()) return it->second;
            const uint32_t idx = static_cast<uint32_t>(dirs.size());
            dirs.push_back(glm::normalize(dirs[a] + dirs[b]));
            edgeMap.emplace(key, idx);
            return idx;
        };

        std::vector<std::array<uint32_t, 3>> next;
        next.reserve(tris.size() * 4);
        for (const auto& t : tris)
        {
            const uint32_t v0 = t[0], v1 = t[1], v2 = t[2];
            const uint32_t v3 = mid(v0, v1), v4 = mid(v1, v2), v5 = mid(v2, v0);
            // Same split as the TypeScript version; preserves winding.
            next.push_back({v0, v3, v5});
            next.push_back({v3, v4, v5});
            next.push_back({v3, v1, v4});
            next.push_back({v5, v4, v2});
        }
        tris.swap(next);
    }

    MeshPrimitive primitive;

    primitive.vertices.reserve(dirs.size());

    for (const glm::vec3& d : dirs)
    {
        gpu::Vertex v;
        v.position = d * radius;
        v.normal   = d;  // exact sphere normal, not a face average
        // Equirectangular UVs. There's a seam at u = 0/1 (triangles straddling
        // it interpolate across the whole texture); fine until textures matter.
        v.uv = glm::vec2(0.5f + std::atan2(d.z, d.x) / glm::two_pi<float>(),
                         0.5f - std::asin(glm::clamp(d.y, -1.0f, 1.0f)) / glm::pi<float>());
        primitive.vertices.push_back(v);
    }

    primitive.indices.reserve(finalTris * 3);
    for (const auto& t : tris)
        primitive.indices.insert(primitive.indices.end(), t.begin(), t.end());

    MeshData mesh;
    mesh.primitives.push_back(std::move(primitive));
    return mesh;
}
