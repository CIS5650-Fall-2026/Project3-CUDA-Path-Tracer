#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace
{

struct PrimitiveBounds
{
    glm::vec3 minimum;
    glm::vec3 maximum;
    glm::vec3 centroid;
};

bool finiteFloat(float value)
{
    return std::isfinite(value);
}

bool finiteVector(const glm::vec3& value)
{
    return finiteFloat(value.x) && finiteFloat(value.y) && finiteFloat(value.z);
}

void includePoint(glm::vec3& minimum, glm::vec3& maximum, const glm::vec3& point)
{
    if (!finiteVector(point))
    {
        throw std::runtime_error("cannot build BVH for non-finite geometry");
    }

    minimum = glm::min(minimum, point);
    maximum = glm::max(maximum, point);
}

glm::vec3 transformPoint(const glm::mat4& transform, const glm::vec3& point)
{
    const glm::vec4 result = transform * glm::vec4(point, 1.0f);
    if (!finiteFloat(result.x) || !finiteFloat(result.y) ||
        !finiteFloat(result.z) || !finiteFloat(result.w) || result.w == 0.0f)
    {
        throw std::runtime_error("cannot build BVH for an invalid transformed point");
    }

    // Scene transforms are affine, but dividing here keeps bounds well-defined
    // if a caller supplies an otherwise valid homogeneous transform.
    return glm::vec3(result) / result.w;
}

PrimitiveBounds boundsForGeom(const Geom& geom)
{
    glm::vec3 minimum(FLT_MAX);
    glm::vec3 maximum(-FLT_MAX);

    if (geom.type == TRIANGLE)
    {
        includePoint(minimum, maximum, geom.vertices[0]);
        includePoint(minimum, maximum, geom.vertices[1]);
        includePoint(minimum, maximum, geom.vertices[2]);
    }
    else if (geom.type == CUBE)
    {
        for (int z = 0; z < 2; ++z)
        {
            for (int y = 0; y < 2; ++y)
            {
                for (int x = 0; x < 2; ++x)
                {
                    const glm::vec3 corner(x == 0 ? -0.5f : 0.5f,
                                           y == 0 ? -0.5f : 0.5f,
                                           z == 0 ? -0.5f : 0.5f);
                    includePoint(minimum, maximum, transformPoint(geom.transform, corner));
                }
            }
        }
    }
    else if (geom.type == SPHERE)
    {
        const glm::vec3 center = transformPoint(geom.transform, glm::vec3(0.0f));

        // An affine image of a radius-r sphere has support radius
        // r * length(row_i(linearTransform)) along world axis i.
        glm::vec3 extent(0.0f);
        for (int row = 0; row < 3; ++row)
        {
            const float squared = geom.transform[0][row] * geom.transform[0][row] +
                                  geom.transform[1][row] * geom.transform[1][row] +
                                  geom.transform[2][row] * geom.transform[2][row];
            if (!finiteFloat(squared) || squared < 0.0f)
            {
                throw std::runtime_error("cannot build BVH for an invalid sphere transform");
            }
            extent[row] = 0.5f * std::sqrt(squared);
        }

        if (!finiteVector(center) || !finiteVector(extent))
        {
            throw std::runtime_error("cannot build BVH for non-finite sphere bounds");
        }
        minimum = center - extent;
        maximum = center + extent;
    }
    else
    {
        throw std::runtime_error("cannot build BVH for an unknown geometry type");
    }

    if (!finiteVector(minimum) || !finiteVector(maximum) ||
        minimum.x > maximum.x || minimum.y > maximum.y || minimum.z > maximum.z)
    {
        throw std::runtime_error("cannot build BVH for invalid primitive bounds");
    }

    // Forward-transformed bounds and inverse-transformed intersection math can
    // differ by several float operations (and by FMA contraction on CUDA).
    // Add a small scale-aware guard band, then step outward four ULPs. This is
    // deliberately a search bound only; exact hit ordering still uses float t.
    const float infinity = std::numeric_limits<float>::infinity();
    for (int axis = 0; axis < 3; ++axis)
    {
        const float magnitude = std::max(1.0f, std::max(std::fabs(minimum[axis]),
                                                        std::fabs(maximum[axis])));
        const float padding = 32.0f * std::numeric_limits<float>::epsilon() * magnitude;
        const float paddedMinimum = minimum[axis] - padding;
        const float paddedMaximum = maximum[axis] + padding;
        minimum[axis] = finiteFloat(paddedMinimum) ? paddedMinimum : -FLT_MAX;
        maximum[axis] = finiteFloat(paddedMaximum) ? paddedMaximum : FLT_MAX;
        for (int step = 0; step < 4; ++step)
        {
            const float nextMinimum = std::nextafter(minimum[axis], -infinity);
            const float nextMaximum = std::nextafter(maximum[axis], infinity);
            if (finiteFloat(nextMinimum))
            {
                minimum[axis] = nextMinimum;
            }
            if (finiteFloat(nextMaximum))
            {
                maximum[axis] = nextMaximum;
            }
        }
    }

    PrimitiveBounds result;
    result.minimum = minimum;
    result.maximum = maximum;
    result.centroid = minimum * 0.5f + maximum * 0.5f;
    if (!finiteVector(result.centroid))
    {
        throw std::runtime_error("cannot build BVH for a non-finite primitive centroid");
    }
    return result;
}

} // namespace

void buildBvh(const std::vector<Geom>& geoms,
              std::vector<BvhNode>& nodes,
              std::vector<int>& primitiveIndices,
              int maxDepth)
{
    if (maxDepth < 1 || maxDepth > 60)
    {
        throw std::invalid_argument("BVH max depth must be in [1, 60]");
    }

    nodes.clear();
    primitiveIndices.clear();
    if (geoms.empty())
    {
        return;
    }

    if (geoms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("too many primitives for the BVH index format");
    }

    std::vector<PrimitiveBounds> primitiveBounds;
    primitiveBounds.reserve(geoms.size());
    for (const Geom& geom : geoms)
    {
        primitiveBounds.push_back(boundsForGeom(geom));
    }

    primitiveIndices.resize(geoms.size());
    std::iota(primitiveIndices.begin(), primitiveIndices.end(), 0);
    nodes.reserve(geoms.size() * 2u - 1u);

    const auto buildRange = [&](const auto& self, int first, int count, int depth) -> int
    {
        BvhNode node;
        node.boundsMin = glm::vec3(FLT_MAX);
        node.boundsMax = glm::vec3(-FLT_MAX);
        glm::vec3 centroidMin(FLT_MAX);
        glm::vec3 centroidMax(-FLT_MAX);

        for (int offset = 0; offset < count; ++offset)
        {
            const PrimitiveBounds& bounds =
                primitiveBounds[static_cast<std::size_t>(primitiveIndices[first + offset])];
            node.boundsMin = glm::min(node.boundsMin, bounds.minimum);
            node.boundsMax = glm::max(node.boundsMax, bounds.maximum);
            centroidMin = glm::min(centroidMin, bounds.centroid);
            centroidMax = glm::max(centroidMax, bounds.centroid);
        }

        const int nodeIndex = static_cast<int>(nodes.size());
        nodes.push_back(node);

        if (count <= 4 || depth >= maxDepth)
        {
            nodes[static_cast<std::size_t>(nodeIndex)].first = first;
            nodes[static_cast<std::size_t>(nodeIndex)].count = count;
            return nodeIndex;
        }

        const glm::vec3 extent = centroidMax - centroidMin;
        int splitAxis = 0;
        if (extent.y > extent.x)
        {
            splitAxis = 1;
        }
        if (extent.z > extent[splitAxis])
        {
            splitAxis = 2;
        }

        // Sorting the whole range costs a little more than nth_element but is
        // deterministic even when many centroids coincide. Primitive id is the
        // final key so repeated builds produce identical index arrays.
        std::sort(primitiveIndices.begin() + first,
                  primitiveIndices.begin() + first + count,
                  [&](int a, int b)
                  {
                      const float ca = primitiveBounds[static_cast<std::size_t>(a)].centroid[splitAxis];
                      const float cb = primitiveBounds[static_cast<std::size_t>(b)].centroid[splitAxis];
                      return ca < cb || (ca == cb && a < b);
                  });

        const int leftCount = count / 2;
        const int rightCount = count - leftCount;
        const int left = self(self, first, leftCount, depth + 1);
        const int right = self(self, first + leftCount, rightCount, depth + 1);

        BvhNode& completed = nodes[static_cast<std::size_t>(nodeIndex)];
        completed.left = left;
        completed.right = right;
        completed.first = 0;
        completed.count = 0;
        return nodeIndex;
    };

    buildRange(buildRange, 0, static_cast<int>(geoms.size()), 0);
}
