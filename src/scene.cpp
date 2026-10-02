#include "scene.h"

#include "gltf_loader.h"
#include "json.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr float kPi = 3.14159265358979323846f;
constexpr std::size_t kMissingIndex = std::numeric_limits<std::size_t>::max();

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

bool finite(float value)
{
    return std::isfinite(static_cast<double>(value));
}

bool finite(const glm::vec2& value)
{
    return finite(value.x) && finite(value.y);
}

bool finite(const glm::vec3& value)
{
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(const glm::mat4& value)
{
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            if (!finite(value[column][row]))
            {
                return false;
            }
        }
    }
    return true;
}

const json& required(const json& object, const char* key, const std::string& context)
{
    if (!object.is_object())
    {
        fail(context + " must be an object");
    }
    const auto iterator = object.find(key);
    if (iterator == object.end())
    {
        fail(context + " is missing required key '" + key + "'");
    }
    return *iterator;
}

std::string readString(const json& value, const std::string& context)
{
    if (!value.is_string())
    {
        fail(context + " must be a string");
    }
    const std::string result = value.get<std::string>();
    if (result.empty())
    {
        fail(context + " must not be empty");
    }
    return result;
}

float readFloat(const json& value, const std::string& context)
{
    if (!value.is_number())
    {
        fail(context + " must be a number");
    }
    const double result = value.get<double>();
    if (!std::isfinite(result) || result < -std::numeric_limits<float>::max() ||
        result > std::numeric_limits<float>::max())
    {
        fail(context + " must be a finite float");
    }
    return static_cast<float>(result);
}

glm::vec3 readVec3(const json& value, const std::string& context)
{
    if (!value.is_array() || value.size() != 3)
    {
        fail(context + " must be an array of exactly three numbers");
    }
    return glm::vec3(readFloat(value[0], context + "[0]"),
                     readFloat(value[1], context + "[1]"),
                     readFloat(value[2], context + "[2]"));
}

int readInt(const json& value, const std::string& context)
{
    if (!value.is_number_integer() && !value.is_number_unsigned())
    {
        fail(context + " must be an integer");
    }
    if (value.is_number_unsigned())
    {
        const std::uint64_t result = value.get<std::uint64_t>();
        if (result > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        {
            fail(context + " is outside the supported integer range");
        }
        return static_cast<int>(result);
    }
    const std::int64_t result = value.get<std::int64_t>();
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
    {
        fail(context + " is outside the supported integer range");
    }
    return static_cast<int>(result);
}

glm::ivec2 readResolution(const json& value, const std::string& context)
{
    if (!value.is_array() || value.size() != 2)
    {
        fail(context + " must be an array of exactly two integers");
    }
    return glm::ivec2(readInt(value[0], context + "[0]"),
                      readInt(value[1], context + "[1]"));
}

fs::path canonicalFile(const fs::path& supplied,
                       const fs::path& relativeTo,
                       const std::string& description)
{
    fs::path candidate = supplied;
    if (candidate.is_relative())
    {
        candidate = relativeTo / candidate;
    }

    std::error_code error;
    const fs::path canonical = fs::canonical(candidate, error);
    if (error)
    {
        fail("Could not resolve " + description + " '" + candidate.string() + "': " +
             error.message());
    }
    if (!fs::is_regular_file(canonical, error) || error)
    {
        fail(description + " is not a regular file: '" + canonical.string() + "'");
    }
    return canonical;
}

void recordAsset(std::vector<std::string>& assets, const fs::path& path)
{
    const std::string canonical = path.string();
    if (std::find(assets.begin(), assets.end(), canonical) == assets.end())
    {
        assets.push_back(canonical);
    }
}

glm::mat4 buildTransform(const glm::vec3& translation,
                         const glm::vec3& rotation,
                         const glm::vec3& scale)
{
    const glm::mat4 identity(1.0f);
    glm::mat4 rotationMatrix = glm::rotate(identity, rotation.x * kPi / 180.0f,
                                            glm::vec3(1.0f, 0.0f, 0.0f));
    rotationMatrix *= glm::rotate(identity, rotation.y * kPi / 180.0f,
                                  glm::vec3(0.0f, 1.0f, 0.0f));
    rotationMatrix *= glm::rotate(identity, rotation.z * kPi / 180.0f,
                                  glm::vec3(0.0f, 0.0f, 1.0f));
    return glm::translate(identity, translation) * rotationMatrix * glm::scale(identity, scale);
}

void initializeTransform(Geom& geom,
                         const glm::vec3& translation,
                         const glm::vec3& rotation,
                         const glm::vec3& scale,
                         const std::string& context)
{
    if (!finite(translation) || !finite(rotation) || !finite(scale))
    {
        fail(context + " has a non-finite transform component");
    }
    if (scale.x == 0.0f || scale.y == 0.0f || scale.z == 0.0f)
    {
        fail(context + " has a singular transform (scale components must be nonzero)");
    }

    geom.translation = translation;
    geom.rotation = rotation;
    geom.scale = scale;
    geom.transform = buildTransform(translation, rotation, scale);

    const float determinant = glm::determinant(glm::mat3(geom.transform));
    if (!finite(determinant) || determinant == 0.0f)
    {
        fail(context + " has a singular or non-finite transform");
    }

    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::transpose(geom.inverseTransform);
    if (!finite(geom.transform) || !finite(geom.inverseTransform) || !finite(geom.invTranspose))
    {
        fail(context + " produces a non-finite transformation matrix");
    }
}

struct StbiPixelsDeleter
{
    void operator()(unsigned char* pixels) const
    {
        stbi_image_free(pixels);
    }
};

int loadTexture(const fs::path& path,
                std::vector<TextureDescriptor>& textures,
                std::vector<glm::vec3>& texels,
                std::unordered_map<std::string, int>& loadedTextures)
{
    const std::string canonicalName = path.string();
    const auto existing = loadedTextures.find(canonicalName);
    if (existing != loadedTextures.end())
    {
        return existing->second;
    }

    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    std::unique_ptr<unsigned char, StbiPixelsDeleter> pixels(
        stbi_load(path.string().c_str(), &width, &height, &sourceChannels, 3));
    if (!pixels)
    {
        const char* reason = stbi_failure_reason();
        fail("Could not decode texture '" + path.string() + "'" +
             (reason != nullptr ? std::string(": ") + reason : std::string()));
    }
    if (width <= 0 || height <= 0)
    {
        fail("Texture '" + path.string() + "' has invalid dimensions");
    }

    const std::uint64_t pixelCount = static_cast<std::uint64_t>(width) *
                                     static_cast<std::uint64_t>(height);
    if (pixelCount > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        texels.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                            static_cast<std::size_t>(pixelCount) ||
        textures.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        fail("Texture '" + path.string() + "' is too large for the scene texture buffers");
    }

    TextureDescriptor descriptor{};
    descriptor.width = width;
    descriptor.height = height;
    descriptor.offset = static_cast<int>(texels.size());
    textures.push_back(descriptor);
    texels.reserve(texels.size() + static_cast<std::size_t>(pixelCount));

    // STB presents the first decoded row at the top. Store row zero as the bottom
    // so that an OBJ/analytic UV of v=0 addresses the lower edge consistently.
    for (int outputY = 0; outputY < height; ++outputY)
    {
        const int sourceY = height - 1 - outputY;
        for (int x = 0; x < width; ++x)
        {
            const std::size_t source =
                (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) * 3;
            glm::vec3 value(static_cast<float>(pixels.get()[source]) / 255.0f,
                            static_cast<float>(pixels.get()[source + 1]) / 255.0f,
                            static_cast<float>(pixels.get()[source + 2]) / 255.0f);
            texels.push_back(value);
        }
    }
    const int textureId = static_cast<int>(textures.size() - 1);
    loadedTextures.emplace(canonicalName, textureId);
    return textureId;
}

double parseObjNumber(const std::string& token,
                      const fs::path& path,
                      std::size_t lineNumber,
                      const char* description)
{
    if (token.empty())
    {
        fail(path.string() + ":" + std::to_string(lineNumber) + ": missing " + description);
    }
    std::size_t cursor = 0;
    if (token[cursor] == '+' || token[cursor] == '-')
    {
        ++cursor;
    }
    bool hasDigits = false;
    while (cursor < token.size() && token[cursor] >= '0' && token[cursor] <= '9')
    {
        hasDigits = true;
        ++cursor;
    }
    if (cursor < token.size() && token[cursor] == '.')
    {
        ++cursor;
        while (cursor < token.size() && token[cursor] >= '0' && token[cursor] <= '9')
        {
            hasDigits = true;
            ++cursor;
        }
    }
    if (!hasDigits)
    {
        cursor = token.size() + 1;
    }
    if (cursor < token.size() && (token[cursor] == 'e' || token[cursor] == 'E'))
    {
        ++cursor;
        if (cursor < token.size() && (token[cursor] == '+' || token[cursor] == '-'))
        {
            ++cursor;
        }
        const std::size_t exponentStart = cursor;
        while (cursor < token.size() && token[cursor] >= '0' && token[cursor] <= '9')
        {
            ++cursor;
        }
        if (cursor == exponentStart)
        {
            cursor = token.size() + 1;
        }
    }
    if (cursor != token.size())
    {
        fail(path.string() + ":" + std::to_string(lineNumber) + ": invalid " + description +
             " '" + token + "'");
    }

    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size() || errno == ERANGE || !std::isfinite(value) ||
        value < -std::numeric_limits<float>::max() || value > std::numeric_limits<float>::max())
    {
        fail(path.string() + ":" + std::to_string(lineNumber) + ": invalid " + description +
             " '" + token + "'");
    }
    return value;
}

long long parseObjIndexNumber(const std::string& token,
                              const fs::path& path,
                              std::size_t lineNumber)
{
    if (token.empty())
    {
        fail(path.string() + ":" + std::to_string(lineNumber) + ": missing face index");
    }
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(token.c_str(), &end, 10);
    if (end != token.c_str() + token.size() || errno == ERANGE)
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": malformed face index '" + token + "'");
    }
    if (value == 0)
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": OBJ index zero is invalid");
    }
    return value;
}

std::size_t resolveObjIndex(long long index,
                            std::size_t count,
                            const char* kind,
                            const fs::path& path,
                            std::size_t lineNumber)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<long long>::max()))
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": OBJ contains too many indexed values");
    }
    const long long resolved = index > 0 ? index - 1 : static_cast<long long>(count) + index;
    if (resolved < 0 || resolved >= static_cast<long long>(count))
    {
        fail(path.string() + ":" + std::to_string(lineNumber) + ": " + kind +
             " index " + std::to_string(index) + " is out of range");
    }
    return static_cast<std::size_t>(resolved);
}

struct ObjCorner
{
    std::size_t position = kMissingIndex;
    std::size_t uv = kMissingIndex;
    std::size_t normal = kMissingIndex;
};

ObjCorner parseObjCorner(const std::string& token,
                         std::size_t positionCount,
                         std::size_t uvCount,
                         std::size_t normalCount,
                         const fs::path& path,
                         std::size_t lineNumber)
{
    ObjCorner corner{};
    const std::size_t firstSlash = token.find('/');
    if (firstSlash == std::string::npos)
    {
        corner.position = resolveObjIndex(parseObjIndexNumber(token, path, lineNumber),
                                          positionCount, "vertex", path, lineNumber);
        return corner;
    }

    if (firstSlash == 0)
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": face corner is missing its vertex index: '" + token + "'");
    }
    const std::size_t secondSlash = token.find('/', firstSlash + 1);
    if (secondSlash != std::string::npos && token.find('/', secondSlash + 1) != std::string::npos)
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": malformed face corner '" + token + "'");
    }

    corner.position = resolveObjIndex(
        parseObjIndexNumber(token.substr(0, firstSlash), path, lineNumber),
        positionCount, "vertex", path, lineNumber);

    if (secondSlash == std::string::npos)
    {
        const std::string uvPart = token.substr(firstSlash + 1);
        if (uvPart.empty())
        {
            fail(path.string() + ":" + std::to_string(lineNumber) +
                 ": malformed face corner '" + token + "'");
        }
        corner.uv = resolveObjIndex(parseObjIndexNumber(uvPart, path, lineNumber), uvCount,
                                    "texture-coordinate", path, lineNumber);
        return corner;
    }

    const std::string uvPart = token.substr(firstSlash + 1, secondSlash - firstSlash - 1);
    const std::string normalPart = token.substr(secondSlash + 1);
    if (normalPart.empty())
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": malformed face corner '" + token + "'");
    }
    if (!uvPart.empty())
    {
        corner.uv = resolveObjIndex(parseObjIndexNumber(uvPart, path, lineNumber), uvCount,
                                    "texture-coordinate", path, lineNumber);
    }
    corner.normal = resolveObjIndex(parseObjIndexNumber(normalPart, path, lineNumber), normalCount,
                                    "normal", path, lineNumber);
    return corner;
}

struct Vec2d
{
    double x = 0.0;
    double y = 0.0;
};

double orient2d(const Vec2d& a, const Vec2d& b, const Vec2d& c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

int signWithTolerance(double value, double tolerance)
{
    return value > tolerance ? 1 : (value < -tolerance ? -1 : 0);
}

bool segmentsIntersect(const Vec2d& a,
                       const Vec2d& b,
                       const Vec2d& c,
                       const Vec2d& d,
                       double orientationTolerance,
                       double coordinateTolerance)
{
    const int abc = signWithTolerance(orient2d(a, b, c), orientationTolerance);
    const int abd = signWithTolerance(orient2d(a, b, d), orientationTolerance);
    const int cda = signWithTolerance(orient2d(c, d, a), orientationTolerance);
    const int cdb = signWithTolerance(orient2d(c, d, b), orientationTolerance);
    if (abc * abd < 0 && cda * cdb < 0)
    {
        return true;
    }

    const auto onSegment = [orientationTolerance, coordinateTolerance](
                               const Vec2d& p, const Vec2d& q, const Vec2d& r) {
        return std::abs(orient2d(p, q, r)) <= orientationTolerance &&
               r.x >= std::min(p.x, q.x) - coordinateTolerance &&
               r.x <= std::max(p.x, q.x) + coordinateTolerance &&
               r.y >= std::min(p.y, q.y) - coordinateTolerance &&
               r.y <= std::max(p.y, q.y) + coordinateTolerance;
    };
    return (abc == 0 && onSegment(a, b, c)) || (abd == 0 && onSegment(a, b, d)) ||
           (cda == 0 && onSegment(c, d, a)) || (cdb == 0 && onSegment(c, d, b));
}

void validatePolygon(const std::vector<ObjCorner>& corners,
                     const std::vector<glm::vec3>& positions,
                     const fs::path& path,
                     std::size_t lineNumber)
{
    const std::string prefix = path.string() + ":" + std::to_string(lineNumber) + ": ";
    if (corners.size() < 3)
    {
        fail(prefix + "a face must contain at least three corners");
    }

    glm::vec3 boundsMin = positions[corners[0].position];
    glm::vec3 boundsMax = boundsMin;
    for (const ObjCorner& corner : corners)
    {
        boundsMin = glm::min(boundsMin, positions[corner.position]);
        boundsMax = glm::max(boundsMax, positions[corner.position]);
    }
    const float scale = glm::length(boundsMax - boundsMin);
    if (!(scale > 0.0f) || !finite(scale))
    {
        fail(prefix + "face is degenerate");
    }
    const float lengthTolerance = scale * 1.0e-6f;
    const float lengthToleranceSquared = lengthTolerance * lengthTolerance;

    for (std::size_t i = 0; i < corners.size(); ++i)
    {
        const glm::vec3& current = positions[corners[i].position];
        const glm::vec3& next = positions[corners[(i + 1) % corners.size()].position];
        if (glm::dot(next - current, next - current) <= lengthToleranceSquared)
        {
            fail(prefix + "face contains a duplicate vertex or degenerate edge");
        }
        for (std::size_t j = i + 1; j < corners.size(); ++j)
        {
            const glm::vec3 difference = positions[corners[j].position] - current;
            if (corners[i].position == corners[j].position ||
                glm::dot(difference, difference) <= lengthToleranceSquared)
            {
                fail(prefix + "face contains duplicate vertices");
            }
        }
    }

    glm::vec3 planeNormal(0.0f);
    float greatestAreaSquared = 0.0f;
    const glm::vec3& origin = positions[corners[0].position];
    for (std::size_t i = 1; i + 1 < corners.size(); ++i)
    {
        const glm::vec3 candidate = glm::cross(positions[corners[i].position] - origin,
                                               positions[corners[i + 1].position] - origin);
        const float areaSquared = glm::dot(candidate, candidate);
        if (areaSquared > greatestAreaSquared)
        {
            greatestAreaSquared = areaSquared;
            planeNormal = candidate;
        }
    }
    const float relativeAreaTolerance = scale * scale * 1.0e-7f;
    if (!(std::sqrt(greatestAreaSquared) > relativeAreaTolerance))
    {
        fail(prefix + "face has zero area");
    }
    planeNormal /= std::sqrt(greatestAreaSquared);

    const float planarTolerance = scale * 1.0e-5f;
    for (const ObjCorner& corner : corners)
    {
        if (std::abs(glm::dot(positions[corner.position] - origin, planeNormal)) > planarTolerance)
        {
            fail(prefix + "face is nonplanar");
        }
    }

    const glm::vec3 absoluteNormal = glm::abs(planeNormal);
    int droppedAxis = 2;
    if (absoluteNormal.x >= absoluteNormal.y && absoluteNormal.x >= absoluteNormal.z)
    {
        droppedAxis = 0;
    }
    else if (absoluteNormal.y >= absoluteNormal.z)
    {
        droppedAxis = 1;
    }

    std::vector<Vec2d> projected;
    projected.reserve(corners.size());
    for (const ObjCorner& corner : corners)
    {
        const glm::vec3& point = positions[corner.position];
        projected.push_back(droppedAxis == 0 ? Vec2d{point.y, point.z}
                                             : (droppedAxis == 1 ? Vec2d{point.x, point.z}
                                                                 : Vec2d{point.x, point.y}));
    }

    const double areaTolerance = static_cast<double>(scale) * static_cast<double>(scale) * 1.0e-7;
    int winding = 0;
    for (std::size_t i = 0; i < projected.size(); ++i)
    {
        const int turn = signWithTolerance(orient2d(projected[i],
                                                    projected[(i + 1) % projected.size()],
                                                    projected[(i + 2) % projected.size()]),
                                           areaTolerance);
        if (turn == 0)
        {
            // A midpoint along a boundary edge is still part of a convex
            // polygon. It is safe when the requested fan does not create a
            // zero-area triangle, which is checked below.
            continue;
        }
        if (winding == 0)
        {
            winding = turn;
        }
        else if (turn != winding)
        {
            fail(prefix + "face is concave");
        }
    }
    if (winding == 0)
    {
        fail(prefix + "face has zero area");
    }

    // The loader deliberately uses a fan rooted at corner zero. Reject a
    // weakly-convex ordering only when that particular fan would contain a
    // degenerate triangle; harmless collinear boundary points remain valid.
    for (std::size_t i = 1; i + 1 < projected.size(); ++i)
    {
        if (signWithTolerance(orient2d(projected[0], projected[i], projected[i + 1]),
                              areaTolerance) == 0)
        {
            fail(prefix + "face fan contains a degenerate triangle");
        }
    }

    for (std::size_t first = 0; first < projected.size(); ++first)
    {
        const std::size_t firstEnd = (first + 1) % projected.size();
        for (std::size_t second = first + 1; second < projected.size(); ++second)
        {
            const std::size_t secondEnd = (second + 1) % projected.size();
            if (first == second || firstEnd == second || secondEnd == first)
            {
                continue;
            }
            if (segmentsIntersect(projected[first], projected[firstEnd],
                                  projected[second], projected[secondEnd], areaTolerance,
                                  static_cast<double>(lengthTolerance)))
            {
                fail(prefix + "face is self-intersecting");
            }
        }
    }
}

glm::vec3 transformPosition(const glm::mat4& transform, const glm::vec3& position)
{
    const glm::vec4 homogeneous = transform * glm::vec4(position, 1.0f);
    return glm::vec3(homogeneous);
}

glm::vec3 transformNormal(const glm::mat4& inverseTranspose,
                          const glm::vec3& normal,
                          const std::string& context)
{
    const glm::vec3 result(inverseTranspose * glm::vec4(normal, 0.0f));
    const float lengthSquared = glm::dot(result, result);
    if (!finite(result) || !finite(lengthSquared) || !(lengthSquared > 0.0f))
    {
        fail(context + " produces an invalid transformed normal");
    }
    return result / std::sqrt(lengthSquared);
}

void appendTriangle(const std::vector<ObjCorner>& face,
                    std::size_t first,
                    std::size_t second,
                    std::size_t third,
                    const std::vector<glm::vec3>& positions,
                    const std::vector<glm::vec2>& uvs,
                    const std::vector<glm::vec3>& normals,
                    const Geom& object,
                    bool mirrored,
                    const fs::path& path,
                    std::size_t lineNumber,
                    std::vector<Geom>& geoms)
{
    ObjCorner triangleCorners[3] = {face[first], face[second], face[third]};
    if (mirrored)
    {
        std::swap(triangleCorners[1], triangleCorners[2]);
    }

    Geom triangle = object;
    triangle.type = TRIANGLE;
    triangle.hasNormals = triangleCorners[0].normal != kMissingIndex;
    triangle.hasUVs = triangleCorners[0].uv != kMissingIndex;

    for (int corner = 0; corner < 3; ++corner)
    {
        triangle.vertices[corner] =
            transformPosition(object.transform, positions[triangleCorners[corner].position]);
        if (!finite(triangle.vertices[corner]))
        {
            fail(path.string() + ":" + std::to_string(lineNumber) +
                 ": transform produces a non-finite triangle vertex");
        }
    }

    const glm::vec3 edge1 = triangle.vertices[1] - triangle.vertices[0];
    const glm::vec3 edge2 = triangle.vertices[2] - triangle.vertices[0];
    const glm::vec3 cross = glm::cross(edge1, edge2);
    const float crossLengthSquared = glm::dot(cross, cross);
    const float edgeProductSquared = glm::dot(edge1, edge1) * glm::dot(edge2, edge2);
    if (!finite(cross) || !finite(crossLengthSquared) || !finite(edgeProductSquared) ||
        !(edgeProductSquared > 0.0f) ||
        crossLengthSquared <= edgeProductSquared * 1.0e-14f)
    {
        fail(path.string() + ":" + std::to_string(lineNumber) +
             ": transform produces a degenerate triangle");
    }
    const glm::vec3 flatNormal = cross / std::sqrt(crossLengthSquared);

    if (triangle.hasNormals)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            triangle.normals[corner] = transformNormal(
                object.invTranspose, normals[triangleCorners[corner].normal],
                path.string() + ":" + std::to_string(lineNumber));
            if (glm::dot(triangle.normals[corner], flatNormal) < 0.0f)
            {
                triangle.normals[corner] = -triangle.normals[corner];
            }
        }
    }
    else
    {
        triangle.normals[0] = flatNormal;
        triangle.normals[1] = flatNormal;
        triangle.normals[2] = flatNormal;
    }

    if (triangle.hasUVs)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            triangle.uvs[corner] = uvs[triangleCorners[corner].uv];
        }
    }
    else
    {
        triangle.uvs[0] = glm::vec2(0.0f, 0.0f);
        triangle.uvs[1] = glm::vec2(1.0f, 0.0f);
        triangle.uvs[2] = glm::vec2(0.0f, 1.0f);
    }
    geoms.push_back(triangle);
}

void loadObj(const fs::path& path, const Geom& object, std::vector<Geom>& geoms)
{
    std::ifstream input(path);
    if (!input)
    {
        fail("Could not open OBJ file '" + path.string() + "'");
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    std::size_t triangleCount = 0;
    std::string line;
    std::size_t lineNumber = 0;
    const bool mirrored = glm::determinant(glm::mat3(object.transform)) < 0.0f;

    while (std::getline(input, line))
    {
        ++lineNumber;
        if (lineNumber == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xef &&
            static_cast<unsigned char>(line[1]) == 0xbb &&
            static_cast<unsigned char>(line[2]) == 0xbf)
        {
            line.erase(0, 3);
        }
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
        {
            line.erase(comment);
        }

        std::istringstream lineStream(line);
        std::string record;
        if (!(lineStream >> record))
        {
            continue;
        }
        std::vector<std::string> tokens;
        std::string token;
        while (lineStream >> token)
        {
            tokens.push_back(token);
        }

        const std::string prefix = path.string() + ":" + std::to_string(lineNumber) + ": ";
        if (record == "v")
        {
            if (tokens.size() != 3 && tokens.size() != 4)
            {
                fail(prefix + "vertex record must have x y z and optional w");
            }
            double x = parseObjNumber(tokens[0], path, lineNumber, "vertex coordinate");
            double y = parseObjNumber(tokens[1], path, lineNumber, "vertex coordinate");
            double z = parseObjNumber(tokens[2], path, lineNumber, "vertex coordinate");
            if (tokens.size() == 4)
            {
                const double w = parseObjNumber(tokens[3], path, lineNumber, "vertex weight");
                if (w == 0.0)
                {
                    fail(prefix + "homogeneous vertex weight must be nonzero");
                }
                x /= w;
                y /= w;
                z /= w;
            }
            const glm::vec3 position(static_cast<float>(x), static_cast<float>(y),
                                     static_cast<float>(z));
            if (!finite(position))
            {
                fail(prefix + "homogeneous vertex conversion is non-finite");
            }
            positions.push_back(position);
        }
        else if (record == "vt")
        {
            if (tokens.size() != 2 && tokens.size() != 3)
            {
                fail(prefix + "texture-coordinate record must have u v and optional w");
            }
            const glm::vec2 uv(
                static_cast<float>(parseObjNumber(tokens[0], path, lineNumber,
                                                  "texture coordinate")),
                static_cast<float>(parseObjNumber(tokens[1], path, lineNumber,
                                                  "texture coordinate")));
            if (tokens.size() == 3)
            {
                (void)parseObjNumber(tokens[2], path, lineNumber, "texture-coordinate w");
            }
            if (!finite(uv))
            {
                fail(prefix + "texture coordinate is non-finite");
            }
            uvs.push_back(uv);
        }
        else if (record == "vn")
        {
            if (tokens.size() != 3)
            {
                fail(prefix + "normal record must have exactly three coordinates");
            }
            glm::vec3 normal(
                static_cast<float>(parseObjNumber(tokens[0], path, lineNumber, "normal coordinate")),
                static_cast<float>(parseObjNumber(tokens[1], path, lineNumber, "normal coordinate")),
                static_cast<float>(parseObjNumber(tokens[2], path, lineNumber, "normal coordinate")));
            const float lengthSquared = glm::dot(normal, normal);
            if (!(lengthSquared > 0.0f) || !finite(lengthSquared) || !finite(normal))
            {
                fail(prefix + "normal must be finite and nonzero");
            }
            normals.push_back(normal / std::sqrt(lengthSquared));
        }
        else if (record == "f")
        {
            if (tokens.size() < 3)
            {
                fail(prefix + "face must have at least three corners");
            }
            std::vector<ObjCorner> face;
            face.reserve(tokens.size());
            for (const std::string& cornerToken : tokens)
            {
                face.push_back(parseObjCorner(cornerToken, positions.size(), uvs.size(),
                                              normals.size(), path, lineNumber));
            }
            bool hasUVs = true;
            bool hasNormals = true;
            for (const ObjCorner& corner : face)
            {
                hasUVs = hasUVs && corner.uv != kMissingIndex;
                hasNormals = hasNormals && corner.normal != kMissingIndex;
            }
            // OBJ permits each corner to select its indices independently. A
            // triangle Geom has one availability flag per attribute, so a
            // partially-authored face deliberately falls back as a whole.
            if (!hasUVs || !hasNormals)
            {
                for (ObjCorner& corner : face)
                {
                    if (!hasUVs)
                    {
                        corner.uv = kMissingIndex;
                    }
                    if (!hasNormals)
                    {
                        corner.normal = kMissingIndex;
                    }
                }
            }

            validatePolygon(face, positions, path, lineNumber);
            for (std::size_t corner = 1; corner + 1 < face.size(); ++corner)
            {
                appendTriangle(face, 0, corner, corner + 1, positions, uvs, normals,
                               object, mirrored, path, lineNumber, geoms);
                ++triangleCount;
            }
        }
        else if (record == "o" || record == "g" || record == "s" ||
                 record == "usemtl" || record == "mtllib")
        {
            // The JSON object selects one material for the entire mesh. These
            // organizational declarations therefore have no scene-level effect.
        }
        else
        {
            fail(prefix + "unsupported OBJ record '" + record + "'");
        }
    }
    if (input.bad())
    {
        fail("I/O error while reading OBJ file '" + path.string() + "'");
    }
    if (triangleCount == 0)
    {
        fail("OBJ file '" + path.string() + "' contains no faces");
    }
}

} // namespace

Scene::Scene(std::string filename)
{
    const std::string suppliedName = filename;
    try
    {
        const fs::path jsonPath = canonicalFile(fs::path(std::move(filename)), fs::current_path(),
                                                "scene file");
        if (jsonPath.extension() != ".json")
        {
            fail("Scene file must use the .json extension: '" + jsonPath.string() + "'");
        }
        loadFromJSON(jsonPath.string());
    }
    catch (const std::runtime_error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        fail("Could not load scene '" + suppliedName + "': " + error.what());
    }
}

SceneView Scene::view() const
{
    SceneView result{};
    result.geoms = geoms.empty() ? nullptr : geoms.data();
    result.geomCount = static_cast<int>(geoms.size());
    result.materials = materials.empty() ? nullptr : materials.data();
    result.materialCount = static_cast<int>(materials.size());
    result.nodes = bvhNodes.empty() ? nullptr : bvhNodes.data();
    result.nodeCount = static_cast<int>(bvhNodes.size());
    result.primitiveIndices = bvhPrimitiveIndices.empty() ? nullptr : bvhPrimitiveIndices.data();
    result.textures = textures.empty() ? nullptr : textures.data();
    result.textureCount = static_cast<int>(textures.size());
    result.texels = texels.empty() ? nullptr : texels.data();
    result.texelCount = static_cast<int>(texels.size());
    return result;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    const fs::path jsonPath(jsonName);
    recordAsset(assets, jsonPath);

    std::ifstream input(jsonPath);
    if (!input)
    {
        fail("Could not open scene file '" + jsonPath.string() + "'");
    }

    json data;
    try
    {
        data = json::parse(input);
    }
    catch (const json::exception& error)
    {
        fail("Could not parse scene JSON '" + jsonPath.string() + "': " + error.what());
    }
    if (!data.is_object())
    {
        fail("Scene JSON root must be an object: '" + jsonPath.string() + "'");
    }
    const fs::path baseDirectory = jsonPath.parent_path();

    if (const auto depth = data.find("BVH_MAX_DEPTH"); depth != data.end())
    {
        bvhMaxDepth = readInt(*depth, "BVH_MAX_DEPTH");
        if (bvhMaxDepth < 1 || bvhMaxDepth > 60)
        {
            fail("BVH_MAX_DEPTH must be between 1 and 60 inclusive");
        }
    }

    const json& materialsData = required(data, "Materials", "scene");
    if (!materialsData.is_object())
    {
        fail("Materials must be an object keyed by material name");
    }
    std::unordered_map<std::string, int> materialIds;
    std::unordered_map<std::string, int> loadedTextures;
    for (const auto& item : materialsData.items())
    {
        const std::string& name = item.key();
        const json& definition = item.value();
        const std::string context = "material '" + name + "'";
        const std::string type = readString(required(definition, "TYPE", context),
                                            context + ".TYPE");
        Material material{};
        material.color = readVec3(required(definition, "RGB", context), context + ".RGB");
        if (material.color.x < 0.0f || material.color.y < 0.0f || material.color.z < 0.0f)
        {
            fail(context + ".RGB components must be nonnegative");
        }
        material.specular.color = material.color;

        if (type == "Diffuse")
        {
        }
        else if (type == "Emitting")
        {
            material.emittance =
                readFloat(required(definition, "EMITTANCE", context), context + ".EMITTANCE");
            if (material.emittance < 0.0f)
            {
                fail(context + ".EMITTANCE must be nonnegative");
            }
        }
        else if (type == "Specular")
        {
            material.hasReflective = 1.0f;
            if (const auto exponent = definition.find("EXPONENT"); exponent != definition.end())
            {
                material.specular.exponent = readFloat(*exponent, context + ".EXPONENT");
                if (material.specular.exponent < 0.0f)
                {
                    fail(context + ".EXPONENT must be nonnegative");
                }
            }
        }
        else if (type == "Glass")
        {
            material.hasRefractive = 1.0f;
            const auto ior = definition.find("IOR");
            const auto longIor = definition.find("INDEX_OF_REFRACTION");
            if (ior != definition.end() && longIor != definition.end())
            {
                fail(context + " must not specify both IOR and INDEX_OF_REFRACTION");
            }
            if (ior == definition.end() && longIor == definition.end())
            {
                fail(context + " is missing required key 'IOR'");
            }
            material.indexOfRefraction = readFloat(
                ior != definition.end() ? *ior : *longIor,
                context + (ior != definition.end() ? ".IOR" : ".INDEX_OF_REFRACTION"));
            if (!(material.indexOfRefraction > 0.0f))
            {
                fail(context + " index of refraction must be positive");
            }
        }
        else
        {
            fail(context + " has unknown TYPE '" + type + "'");
        }

        if (const auto texture = definition.find("TEXTURE"); texture != definition.end())
        {
            const fs::path texturePath = canonicalFile(
                fs::path(readString(*texture, context + ".TEXTURE")), baseDirectory,
                "texture for " + context);
            recordAsset(assets, texturePath);
            material.textureId = loadTexture(texturePath, textures, texels, loadedTextures);
        }
        if (const auto procedural = definition.find("PROCEDURAL");
            procedural != definition.end())
        {
            const std::string value = readString(*procedural, context + ".PROCEDURAL");
            if (value == "none" || value == "None" || value == "NONE")
            {
                material.procedural = 0;
            }
            else if (value == "checker" || value == "Checker" || value == "CHECKER")
            {
                material.procedural = 1;
            }
            else if (value == "marble" || value == "Marble" || value == "MARBLE")
            {
                material.procedural = 2;
            }
            else if (value == "cells" || value == "Cells" || value == "CELLS")
            {
                material.procedural = 3;
            }
            else
            {
                fail(context +
                     ".PROCEDURAL supports only 'none', 'checker', 'marble', or 'cells'");
            }
        }
        if (const auto scale = definition.find("CHECKER_SCALE"); scale != definition.end())
        {
            if (material.procedural != 1)
            {
                fail(context + ".CHECKER_SCALE requires PROCEDURAL 'checker'");
            }
            material.checkerScale = readFloat(*scale, context + ".CHECKER_SCALE");
            if (!(material.checkerScale > 0.0f))
            {
                fail(context + ".CHECKER_SCALE must be positive");
            }
        }
        if (const auto bump = definition.find("BUMP"); bump != definition.end())
        {
            const fs::path bumpPath = canonicalFile(
                fs::path(readString(*bump, context + ".BUMP")), baseDirectory,
                "bump texture for " + context);
            recordAsset(assets, bumpPath);
            material.bumpTextureId = loadTexture(bumpPath, textures, texels, loadedTextures);
            const auto strength = definition.find("BUMP_STRENGTH");
            const auto strengthAlias = definition.find("STRENGTH");
            if (strength != definition.end() && strengthAlias != definition.end())
            {
                const float primaryValue = readFloat(*strength, context + ".BUMP_STRENGTH");
                const float aliasValue = readFloat(*strengthAlias, context + ".STRENGTH");
                if (primaryValue != aliasValue)
                {
                    fail(context +
                         " specifies conflicting BUMP_STRENGTH and STRENGTH values");
                }
                material.bumpStrength = primaryValue;
            }
            else if (strength != definition.end() || strengthAlias != definition.end())
            {
                material.bumpStrength = readFloat(
                    strength != definition.end() ? *strength : *strengthAlias,
                    context + (strength != definition.end() ? ".BUMP_STRENGTH"
                                                             : ".STRENGTH"));
            }
        }
        else if (definition.contains("STRENGTH") || definition.contains("BUMP_STRENGTH"))
        {
            fail(context + ".STRENGTH requires BUMP");
        }

        if (materials.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            fail("Scene contains too many materials");
        }
        materialIds.emplace(name, static_cast<int>(materials.size()));
        materials.push_back(material);
    }

    const json& objectsData = required(data, "Objects", "scene");
    if (!objectsData.is_array())
    {
        fail("Objects must be an array");
    }
    for (std::size_t objectIndex = 0; objectIndex < objectsData.size(); ++objectIndex)
    {
        const json& definition = objectsData[objectIndex];
        const std::string context = "object[" + std::to_string(objectIndex) + "]";
        const std::string type = readString(required(definition, "TYPE", context),
                                            context + ".TYPE");
        const std::string materialName = readString(required(definition, "MATERIAL", context),
                                                    context + ".MATERIAL");
        const auto material = materialIds.find(materialName);
        if (material == materialIds.end())
        {
            fail(context + " references unknown material '" + materialName + "'");
        }

        Geom object{};
        object.materialid = material->second;
        if (const auto motion = definition.find("MOTION_TRANSLATION");
            motion != definition.end())
        {
            object.motionTranslation = readVec3(*motion, context + ".MOTION_TRANSLATION");
        }
        initializeTransform(object,
                            readVec3(required(definition, "TRANS", context), context + ".TRANS"),
                            readVec3(required(definition, "ROTAT", context), context + ".ROTAT"),
                            readVec3(required(definition, "SCALE", context), context + ".SCALE"),
                            context);

        if (type == "cube")
        {
            object.type = CUBE;
            geoms.push_back(object);
        }
        else if (type == "sphere")
        {
            object.type = SPHERE;
            geoms.push_back(object);
        }
        else if (type == "obj")
        {
            const fs::path objPath = canonicalFile(
                fs::path(readString(required(definition, "FILE", context), context + ".FILE")),
                baseDirectory, "OBJ mesh for " + context);
            recordAsset(assets, objPath);
            loadObj(objPath, object, geoms);
        }
        else if (type == "gltf")
        {
            const fs::path modelPath = canonicalFile(
                fs::path(readString(required(definition, "FILE", context), context + ".FILE")),
                baseDirectory, "glTF mesh for " + context);
            recordAsset(assets, modelPath);
            gltf::LoadResult loaded = gltf::load(modelPath, object);
            for (const fs::path& asset : loaded.externalAssets)
            {
                recordAsset(assets, asset);
            }
            const std::size_t maximumPrimitiveCount = static_cast<std::size_t>(
                std::numeric_limits<int>::max());
            if (geoms.size() > maximumPrimitiveCount
                || loaded.triangles.size() > maximumPrimitiveCount - geoms.size())
            {
                fail("Scene contains too many primitives");
            }
            geoms.insert(geoms.end(),
                         std::make_move_iterator(loaded.triangles.begin()),
                         std::make_move_iterator(loaded.triangles.end()));
        }
        else
        {
            fail(context + " has unknown TYPE '" + type + "'");
        }
    }

    const json& cameraData = required(data, "Camera", "scene");
    Camera& camera = state.camera;
    camera.resolution = readResolution(required(cameraData, "RES", "Camera"), "Camera.RES");
    if (camera.resolution.x <= 0 || camera.resolution.y <= 0)
    {
        fail("Camera.RES dimensions must be positive");
    }
    const float fovy = readFloat(required(cameraData, "FOVY", "Camera"), "Camera.FOVY");
    if (fovy < 0.0f || fovy > 89.9f)
    {
        fail("Camera.FOVY must be between 0 and 89.9 degrees inclusive");
    }

    const int iterations = readInt(required(cameraData, "ITERATIONS", "Camera"),
                                   "Camera.ITERATIONS");
    if (iterations <= 0)
    {
        fail("Camera.ITERATIONS must be positive");
    }
    state.iterations = static_cast<unsigned int>(iterations);
    state.traceDepth = readInt(required(cameraData, "DEPTH", "Camera"), "Camera.DEPTH");
    if (state.traceDepth <= 0)
    {
        fail("Camera.DEPTH must be positive");
    }
    state.imageName = readString(required(cameraData, "FILE", "Camera"), "Camera.FILE");
    camera.position = readVec3(required(cameraData, "EYE", "Camera"), "Camera.EYE");
    camera.lookAt = readVec3(required(cameraData, "LOOKAT", "Camera"), "Camera.LOOKAT");
    const glm::vec3 suppliedUp = readVec3(required(cameraData, "UP", "Camera"), "Camera.UP");
    camera.shutterOpen = 0.0f;
    camera.shutterClose = 1.0f;
    if (const auto shutter = cameraData.find("SHUTTER"); shutter != cameraData.end())
    {
        if (!shutter->is_array() || shutter->size() != 2)
        {
            fail("Camera.SHUTTER must be an array of exactly two numbers");
        }
        camera.shutterOpen = readFloat((*shutter)[0], "Camera.SHUTTER[0]");
        camera.shutterClose = readFloat((*shutter)[1], "Camera.SHUTTER[1]");
        // Enforce the schema before narrowing to binary32 so a tiny negative,
        // just-over-one, or narrowly reversed JSON value cannot round onto a
        // valid float boundary.
        const double shutterOpen = (*shutter)[0].get<double>();
        const double shutterClose = (*shutter)[1].get<double>();
        if (shutterOpen < 0.0 || shutterOpen > shutterClose ||
            shutterClose > 1.0)
        {
            fail("Camera.SHUTTER must satisfy 0 <= open <= close <= 1");
        }
    }

    const glm::vec3 viewVector = camera.lookAt - camera.position;
    const float viewLengthSquared = glm::dot(viewVector, viewVector);
    const float upLengthSquared = glm::dot(suppliedUp, suppliedUp);
    if (!finite(viewVector) || !finite(viewLengthSquared) || !(viewLengthSquared > 0.0f))
    {
        fail("Camera.EYE and Camera.LOOKAT must be different");
    }
    if (!finite(upLengthSquared) || !(upLengthSquared > 0.0f))
    {
        fail("Camera.UP must be nonzero");
    }
    camera.view = viewVector / std::sqrt(viewLengthSquared);
    const glm::vec3 rightVector = glm::cross(camera.view, suppliedUp);
    const float rightLengthSquared = glm::dot(rightVector, rightVector);
    if (!finite(camera.view) || !finite(rightVector) || !finite(rightLengthSquared) ||
        !(rightLengthSquared > 1.0e-12f * upLengthSquared))
    {
        fail("Camera.UP must not be collinear with the viewing direction");
    }
    camera.right = rightVector / std::sqrt(rightLengthSquared);
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    if (!finite(camera.right) || !finite(camera.up))
    {
        fail("Camera basis is non-finite");
    }

    camera.aperture = 0.0f;
    if (const auto aperture = cameraData.find("APERTURE"); aperture != cameraData.end())
    {
        camera.aperture = readFloat(*aperture, "Camera.APERTURE");
        if (camera.aperture < 0.0f)
        {
            fail("Camera.APERTURE must be nonnegative");
        }
    }
    camera.focusDistance = std::sqrt(viewLengthSquared);
    const auto focus = cameraData.find("FOCUS");
    const auto longFocus = cameraData.find("FOCUS_DISTANCE");
    if (focus != cameraData.end() && longFocus != cameraData.end())
    {
        fail("Camera must not specify both FOCUS and FOCUS_DISTANCE");
    }
    if (focus != cameraData.end() || longFocus != cameraData.end())
    {
        camera.focusDistance = readFloat(focus != cameraData.end() ? *focus : *longFocus,
                                         focus != cameraData.end() ? "Camera.FOCUS"
                                                                   : "Camera.FOCUS_DISTANCE");
        if (!(camera.focusDistance > 0.0f))
        {
            fail("Camera.FOCUS must be positive");
        }
    }

    // FOVY is the starter's half-angle convention and is intentionally not halved here.
    const float yScaled = std::tan(fovy * kPi / 180.0f);
    const float xScaled = yScaled * static_cast<float>(camera.resolution.x) /
                          static_cast<float>(camera.resolution.y);
    const float fovx = std::atan(xScaled) * 180.0f / kPi;
    if (!finite(yScaled) || !finite(xScaled) || !finite(fovx))
    {
        fail("Camera FOV produces non-finite projection values");
    }
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(2.0f * xScaled / static_cast<float>(camera.resolution.x),
                                   2.0f * yScaled / static_cast<float>(camera.resolution.y));
    if (!finite(camera.fov) || !finite(camera.pixelLength))
    {
        fail("Camera projection is non-finite");
    }

    const std::uint64_t pixelCount = static_cast<std::uint64_t>(camera.resolution.x) *
                                     static_cast<std::uint64_t>(camera.resolution.y);
    if (pixelCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        pixelCount > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        fail("Camera resolution is too large");
    }
    state.image.assign(static_cast<std::size_t>(pixelCount), glm::vec3(0.0f));

    if (geoms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        fail("Scene contains too many primitives");
    }
    buildBvh(geoms, bvhNodes, bvhPrimitiveIndices, bvhMaxDepth);
}
