#pragma once

#include "sceneStructs.h"

#include <filesystem>
#include <vector>

namespace gltf
{

// The loader deliberately exposes no TinyGLTF types. Geometry is flattened to
// renderer-native world-space triangles, while externalAssets contains the
// canonical local files whose bytes participate in checkpoint identity.
struct LoadResult
{
    std::vector<Geom> triangles;
    std::vector<std::filesystem::path> externalAssets;
};

LoadResult load(const std::filesystem::path& canonicalModelPath,
                const Geom& sceneObject);

} // namespace gltf
