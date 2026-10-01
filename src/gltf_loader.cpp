#include "gltf_loader.h"

#include "json.hpp"

// This is the single TinyGLTF implementation translation unit. Reuse the
// starter's nlohmann JSON instead of allowing TinyGLTF to provide a duplicate
// copy. Native glTF image pixels and materials are intentionally not imported,
// so a local no-op image callback avoids both another STB implementation and
// TinyGLTF's optional 16-bit STB API, which the unchanged starter STB predates.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <tiny_gltf.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gltf
{
namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr std::size_t kMaximumHierarchyDepth = 4096;
constexpr std::uint32_t kGlbMagic = 0x46546c67U;
constexpr std::uint32_t kGlbJsonChunk = 0x4e4f534aU;
constexpr std::uint32_t kGlbBinChunk = 0x004e4942U;

[[noreturn]] void fail(const fs::path& path, const std::string& message)
{
    throw std::runtime_error("glTF '" + path.string() + "': " + message);
}

bool finite(double value)
{
    return std::isfinite(value);
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
            if (!finite(value[column][row])) return false;
        }
    }
    return true;
}

bool checkedAdd(std::size_t first, std::size_t second, std::size_t* result)
{
    if (first > std::numeric_limits<std::size_t>::max() - second) return false;
    *result = first + second;
    return true;
}

bool checkedMultiply(std::size_t first, std::size_t second,
                     std::size_t* result)
{
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
    {
        return false;
    }
    *result = first * second;
    return true;
}

std::uint32_t readU32(const unsigned char* bytes)
{
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8U)
        | (static_cast<std::uint32_t>(bytes[2]) << 16U)
        | (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint16_t readU16(const unsigned char* bytes)
{
    return static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

float readFloat32(const unsigned char* bytes)
{
    const std::uint32_t bits = readU32(bytes);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<unsigned char> readFile(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail(path, "could not open model file");
    const std::streamoff end = input.tellg();
    if (end <= 0) fail(path, "model file is empty");
    if (static_cast<std::uint64_t>(end)
        > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max()))
    {
        fail(path, "model file exceeds the parser's supported size");
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes.data()), end);
    if (!input || input.gcount() != end) fail(path, "could not read model file");
    return bytes;
}

std::size_t unsignedValue(const json& value, const fs::path& path,
                          const std::string& context)
{
    if (!value.is_number_unsigned() && !value.is_number_integer())
    {
        fail(path, context + " must be a nonnegative integer");
    }
    if (value.is_number_unsigned())
    {
        const std::uint64_t result = value.get<std::uint64_t>();
        if (result > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()))
        {
            fail(path, context + " exceeds the supported size range");
        }
        return static_cast<std::size_t>(result);
    }
    else
    {
        const std::int64_t signedValue = value.get<std::int64_t>();
        if (signedValue < 0)
        {
            fail(path, context + " must be a nonnegative integer");
        }
        return static_cast<std::size_t>(signedValue);
    }
}

int indexValue(const json& value, const fs::path& path,
               const std::string& context)
{
    const std::size_t result = unsignedValue(value, path, context);
    if (result > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        fail(path, context + " exceeds the supported index range");
    }
    return static_cast<int>(result);
}

double finiteNumber(const json& value, const fs::path& path,
                    const std::string& context)
{
    if (!value.is_number()) fail(path, context + " must be a number");
    const double result = value.get<double>();
    if (!finite(result)
        || result < -static_cast<double>(std::numeric_limits<float>::max())
        || result > static_cast<double>(std::numeric_limits<float>::max()))
    {
        fail(path, context + " must be finite and representable as binary32");
    }
    return result;
}

const json* optionalMember(const json& object, const char* name,
                           const fs::path& path,
                           const std::string& context)
{
    if (!object.is_object()) fail(path, context + " must be an object");
    const auto iterator = object.find(name);
    return iterator == object.end() ? nullptr : &*iterator;
}

const json& requiredMember(const json& object, const char* name,
                           const fs::path& path,
                           const std::string& context)
{
    const json* value = optionalMember(object, name, path, context);
    if (value == nullptr) fail(path, context + " is missing '" + name + "'");
    return *value;
}

const json& optionalArray(const json& root, const char* name,
                          const fs::path& path)
{
    static const json empty = json::array();
    const json* value = optionalMember(root, name, path, "root");
    if (value == nullptr) return empty;
    if (!value->is_array()) fail(path, std::string(name) + " must be an array");
    return *value;
}

int hexNibble(char value)
{
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

std::string decodeExternalUri(const std::string& uri, const fs::path& path,
                              const std::string& context)
{
    if (uri.empty()) fail(path, context + " URI must not be empty");
    if (uri.find('?') != std::string::npos || uri.find('#') != std::string::npos)
    {
        fail(path, context + " URI query and fragment components are unsupported");
    }
    std::string decoded;
    decoded.reserve(uri.size());
    for (std::size_t index = 0; index < uri.size(); ++index)
    {
        unsigned char value = static_cast<unsigned char>(uri[index]);
        if (value == '%')
        {
            if (index + 2 >= uri.size())
            {
                fail(path, context + " URI contains an incomplete percent escape");
            }
            const int high = hexNibble(uri[index + 1]);
            const int low = hexNibble(uri[index + 2]);
            if (high < 0 || low < 0)
            {
                fail(path, context + " URI contains an invalid percent escape");
            }
            value = static_cast<unsigned char>((high << 4) | low);
            index += 2;
        }
        if (value == 0 || value < 0x20U || value == 0x7fU)
        {
            fail(path, context + " URI contains a forbidden control byte");
        }
        decoded.push_back(static_cast<char>(value));
    }
    if (decoded.empty()) fail(path, context + " URI decodes to an empty path");
    if (decoded.find(':') != std::string::npos)
    {
        fail(path, context + " URI schemes, including file and remote URIs, are unsupported");
    }
    const fs::path decodedPath(decoded);
    if (decodedPath.is_absolute() || decodedPath.has_root_directory()
        || decodedPath.has_root_name() || decoded.rfind("//", 0) == 0
        || decoded.rfind("\\\\", 0) == 0)
    {
        fail(path, context + " URI must be relative to the model file");
    }
    return decoded;
}

bool strictUriDecode(const std::string& input, std::string* output,
                     void* userData)
{
    auto* error = static_cast<std::string*>(userData);
    try
    {
        *output = decodeExternalUri(input, fs::path("<parser>"), "external");
        return true;
    }
    catch (const std::exception& exception)
    {
        if (error != nullptr) *error = exception.what();
        return false;
    }
}

bool base64Character(char value)
{
    return (value >= 'A' && value <= 'Z')
        || (value >= 'a' && value <= 'z')
        || (value >= '0' && value <= '9') || value == '+' || value == '/';
}

int base64Value(char value)
{
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

void validateDataUri(const std::string& uri, const fs::path& path,
                     const std::string& context)
{
    if (!tinygltf::IsDataURI(uri))
    {
        fail(path, context + " uses an unsupported or malformed data URI");
    }
    const std::size_t comma = uri.find(',');
    if (comma == std::string::npos || comma + 1 >= uri.size()
        || uri.substr(0, comma).find(";base64") == std::string::npos)
    {
        fail(path, context + " data URI must contain a nonempty base64 payload");
    }
    const std::string payload = uri.substr(comma + 1);
    if (payload.size() % 4 != 0)
    {
        fail(path, context + " data URI base64 length is invalid");
    }
    std::size_t padding = 0;
    if (!payload.empty() && payload.back() == '=') ++padding;
    if (payload.size() >= 2 && payload[payload.size() - 2] == '=') ++padding;
    if (padding > 2) fail(path, context + " data URI base64 padding is invalid");
    for (std::size_t index = 0; index < payload.size(); ++index)
    {
        if (index >= payload.size() - padding)
        {
            if (payload[index] != '=')
                fail(path, context + " data URI base64 padding is invalid");
        }
        else if (!base64Character(payload[index]))
        {
            fail(path, context + " data URI contains invalid base64 data");
        }
    }
    if (padding == 2 && (base64Value(payload[payload.size() - 3]) & 0x0f) != 0)
        fail(path, context + " data URI has non-canonical base64 padding bits");
    if (padding == 1 && (base64Value(payload[payload.size() - 2]) & 0x03) != 0)
        fail(path, context + " data URI has non-canonical base64 padding bits");
}

fs::path canonicalExternalAsset(const fs::path& modelPath,
                                const std::string& uri,
                                const std::string& context)
{
    const std::string decoded = decodeExternalUri(uri, modelPath, context);
    std::error_code error;
    const fs::path candidate = modelPath.parent_path() / fs::path(decoded);
    const fs::path canonical = fs::canonical(candidate, error);
    if (error || !fs::is_regular_file(canonical, error) || error)
    {
        fail(modelPath, context + " external asset could not be resolved: '"
            + candidate.string() + "'");
    }
    const std::uintmax_t size = fs::file_size(canonical, error);
    if (error || size == 0)
    {
        fail(modelPath, context + " external asset is empty or unreadable: '"
            + canonical.string() + "'");
    }
    return canonical;
}

void appendUniqueAsset(std::vector<fs::path>* assets, const fs::path& asset)
{
    const std::string name = asset.string();
    for (const fs::path& existing : *assets)
    {
        if (existing.string() == name) return;
    }
    assets->push_back(asset);
}

struct ParsedDocument
{
    json root;
    std::vector<unsigned char> fileBytes;
    std::string jsonText;
    bool binary = false;
    bool hasBinaryChunk = false;
    std::size_t binaryChunkLength = 0;
};

ParsedDocument parseDocument(const fs::path& path)
{
    ParsedDocument document;
    document.fileBytes = readFile(path);
    const std::string extension = path.extension().string();
    if (extension == ".gltf")
    {
        document.jsonText.assign(document.fileBytes.begin(),
                                 document.fileBytes.end());
    }
    else if (extension == ".glb")
    {
        document.binary = true;
        if (document.fileBytes.size() < 20)
            fail(path, "GLB header or JSON chunk is truncated");
        const std::uint32_t magic = readU32(document.fileBytes.data());
        const std::uint32_t version = readU32(document.fileBytes.data() + 4);
        const std::uint32_t declaredLength = readU32(document.fileBytes.data() + 8);
        if (magic != kGlbMagic) fail(path, "GLB magic is invalid");
        if (version != 2U) fail(path, "only GLB version 2 is supported");
        if (declaredLength != document.fileBytes.size())
            fail(path, "GLB declared length does not match the file size");

        std::size_t cursor = 12;
        int chunkIndex = 0;
        bool sawJson = false;
        bool sawBin = false;
        while (cursor < document.fileBytes.size())
        {
            std::size_t headerEnd = 0;
            if (!checkedAdd(cursor, 8, &headerEnd)
                || headerEnd > document.fileBytes.size())
                fail(path, "GLB chunk header is truncated");
            const std::uint32_t chunkLength = readU32(document.fileBytes.data() + cursor);
            const std::uint32_t chunkType = readU32(document.fileBytes.data() + cursor + 4);
            if ((chunkLength & 3U) != 0U)
                fail(path, "GLB chunk length must be four-byte aligned");
            const std::size_t payload = cursor + 8;
            std::size_t next = 0;
            if (!checkedAdd(payload, chunkLength, &next)
                || next > document.fileBytes.size())
                fail(path, "GLB chunk extends beyond the file");
            if (chunkIndex == 0)
            {
                if (chunkType != kGlbJsonChunk || chunkLength == 0)
                    fail(path, "GLB first chunk must be nonempty JSON");
                document.jsonText.assign(
                    reinterpret_cast<const char*>(document.fileBytes.data() + payload),
                    chunkLength);
                sawJson = true;
            }
            else if (chunkIndex == 1 && chunkType == kGlbBinChunk)
            {
                sawBin = true;
                document.hasBinaryChunk = true;
                document.binaryChunkLength = chunkLength;
            }
            else
            {
                fail(path, "unsupported extra or out-of-order GLB chunk");
            }
            cursor = next;
            ++chunkIndex;
        }
        if (!sawJson || chunkIndex > 2 || (chunkIndex == 2 && !sawBin))
            fail(path, "GLB chunk layout is invalid");
    }
    else
    {
        fail(path, "model file must use the .gltf or .glb extension");
    }

    try
    {
        document.root = json::parse(document.jsonText);
    }
    catch (const json::exception& exception)
    {
        fail(path, std::string("could not parse model JSON: ") + exception.what());
    }
    if (!document.root.is_object()) fail(path, "model JSON root must be an object");
    return document;
}

void validateRawVector(const json& object, const char* name,
                       std::size_t expectedSize, const fs::path& path,
                       const std::string& context)
{
    const json* value = optionalMember(object, name, path, context);
    if (value == nullptr) return;
    if (!value->is_array() || value->size() != expectedSize)
    {
        fail(path, context + "." + name + " must contain exactly "
            + std::to_string(expectedSize) + " numbers");
    }
    for (std::size_t index = 0; index < value->size(); ++index)
    {
        (void)finiteNumber((*value)[index], path, context + "." + name
            + "[" + std::to_string(index) + "]");
    }
}

std::size_t rawComponentWidth(int componentType, const fs::path& path,
                              const std::string& context)
{
    switch (componentType)
    {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return 4;
        default:
            fail(path, context + " uses an unsupported component type");
    }
}

struct RawAccessorShape
{
    std::size_t rows = 1;
    std::size_t columns = 1;
    bool matrix = false;
};

RawAccessorShape rawAccessorShape(const std::string& type,
                                  const fs::path& path,
                                  const std::string& context)
{
    if (type == "SCALAR") return {};
    if (type == "VEC2") return {2, 1, false};
    if (type == "VEC3") return {3, 1, false};
    if (type == "VEC4") return {4, 1, false};
    if (type == "MAT2") return {2, 2, true};
    if (type == "MAT3") return {3, 3, true};
    if (type == "MAT4") return {4, 4, true};
    fail(path, context + " uses an unsupported accessor shape");
}

std::size_t rawElementSize(std::size_t componentWidth,
                           const RawAccessorShape& shape,
                           const fs::path& path,
                           const std::string& context)
{
    std::size_t columnSize = 0;
    if (!checkedMultiply(componentWidth, shape.rows, &columnSize))
        fail(path, context + " accessor element size overflows");
    if (shape.matrix)
    {
        // glTF pads each matrix column to a four-byte boundary when its
        // components are one or two bytes wide.
        const std::size_t remainder = columnSize % 4;
        if (remainder != 0 && !checkedAdd(columnSize, 4 - remainder, &columnSize))
            fail(path, context + " accessor matrix padding overflows");
    }
    std::size_t result = 0;
    if (!checkedMultiply(columnSize, shape.columns, &result))
        fail(path, context + " accessor element size overflows");
    return result;
}

void validateRawModel(const ParsedDocument& document, const fs::path& path,
                      std::vector<fs::path>* assets)
{
    const json& root = document.root;
    const json& asset = requiredMember(root, "asset", path, "root");
    const json& version = requiredMember(asset, "version", path, "asset");
    if (!version.is_string() || version.get<std::string>() != "2.0")
        fail(path, "asset.version must be exactly '2.0'");
    if (const json* minimumVersion = optionalMember(asset, "minVersion", path,
                                                     "asset"))
    {
        if (!minimumVersion->is_string()
            || minimumVersion->get<std::string>() != "2.0")
            fail(path, "asset.minVersion must be '2.0' when present");
    }

    const json& requiredExtensions = optionalArray(root, "extensionsRequired", path);
    if (!requiredExtensions.empty())
        fail(path, "required extensions are unsupported");
    if (!optionalArray(root, "animations", path).empty())
        fail(path, "animations are unsupported");
    if (!optionalArray(root, "skins", path).empty())
        fail(path, "skins are unsupported");

    const json& buffers = optionalArray(root, "buffers", path);
    std::vector<std::size_t> bufferLengths;
    bufferLengths.reserve(buffers.size());
    std::size_t binaryBackedBuffers = 0;
    for (std::size_t index = 0; index < buffers.size(); ++index)
    {
        const json& buffer = buffers[index];
        const std::string context = "buffers[" + std::to_string(index) + "]";
        const std::size_t byteLength = unsignedValue(
            requiredMember(buffer, "byteLength", path, context), path,
            context + ".byteLength");
        if (byteLength == 0) fail(path, context + ".byteLength must be positive");
        bufferLengths.push_back(byteLength);
        const json* uriValue = optionalMember(buffer, "uri", path, context);
        if (uriValue == nullptr)
        {
            if (!document.binary)
                fail(path, context + ".uri is required for ASCII glTF");
            if (index != 0 || binaryBackedBuffers != 0)
                fail(path, "only buffers[0] may omit uri and use the GLB BIN chunk");
            ++binaryBackedBuffers;
        }
        else
        {
            if (!uriValue->is_string()) fail(path, context + ".uri must be a string");
            const std::string uri = uriValue->get<std::string>();
            if (uri.rfind("data:", 0) == 0)
            {
                validateDataUri(uri, path, context);
            }
            else
            {
                appendUniqueAsset(assets,
                    canonicalExternalAsset(path, uri, context));
            }
        }
    }

    if (document.binary)
    {
        if ((binaryBackedBuffers == 1) != document.hasBinaryChunk)
            fail(path, "GLB BIN chunk presence does not match its URI-less buffer");
        if (binaryBackedBuffers == 1)
        {
            std::size_t paddedMaximum = 0;
            if (!checkedAdd(bufferLengths[0], 3, &paddedMaximum)
                || document.binaryChunkLength < bufferLengths[0]
                || document.binaryChunkLength > paddedMaximum)
                fail(path, "GLB BIN chunk must match buffers[0].byteLength with at most three padding bytes");
        }
    }

    const json& views = optionalArray(root, "bufferViews", path);
    const json& images = optionalArray(root, "images", path);
    for (std::size_t index = 0; index < images.size(); ++index)
    {
        const json& image = images[index];
        const std::string context = "images[" + std::to_string(index) + "]";
        const json* uriValue = optionalMember(image, "uri", path, context);
        const json* viewValue = optionalMember(image, "bufferView", path, context);
        if ((uriValue == nullptr) == (viewValue == nullptr))
            fail(path, context + " must specify exactly one of uri or bufferView");
        if (uriValue != nullptr)
        {
            if (!uriValue->is_string()) fail(path, context + ".uri must be a string");
            const std::string uri = uriValue->get<std::string>();
            if (uri.rfind("data:", 0) == 0) validateDataUri(uri, path, context);
            else appendUniqueAsset(assets,
                canonicalExternalAsset(path, uri, context));
        }
        else
        {
            const int view = indexValue(*viewValue, path,
                                        context + ".bufferView");
            if (static_cast<std::size_t>(view) >= views.size())
                fail(path, context + ".bufferView is out of range");
        }
    }

    struct RawView
    {
        std::size_t byteOffset = 0;
        std::size_t byteLength = 0;
        std::size_t byteStride = 0;
    };
    std::vector<RawView> rawViews;
    rawViews.reserve(views.size());
    for (std::size_t index = 0; index < views.size(); ++index)
    {
        const json& view = views[index];
        const std::string context = "bufferViews[" + std::to_string(index) + "]";
        const int bufferIndex = indexValue(
            requiredMember(view, "buffer", path, context), path,
            context + ".buffer");
        if (static_cast<std::size_t>(bufferIndex) >= buffers.size())
            fail(path, context + ".buffer is out of range");
        const std::size_t length = unsignedValue(
            requiredMember(view, "byteLength", path, context), path,
            context + ".byteLength");
        if (length == 0) fail(path, context + ".byteLength must be positive");
        std::size_t offsetValue = 0;
        if (const json* offset = optionalMember(view, "byteOffset", path, context))
            offsetValue = unsignedValue(*offset, path, context + ".byteOffset");
        std::size_t end = 0;
        if (!checkedAdd(offsetValue, length, &end)
            || end > bufferLengths[static_cast<std::size_t>(bufferIndex)])
            fail(path, context + " range exceeds its declared buffer length");
        std::size_t strideValue = 0;
        if (const json* stride = optionalMember(view, "byteStride", path, context))
        {
            strideValue = unsignedValue(*stride, path,
                                        context + ".byteStride");
            if (strideValue < 4 || strideValue > 252 || strideValue % 4 != 0)
                fail(path, context + ".byteStride must be a multiple of four in [4,252]");
        }
        rawViews.push_back({offsetValue, length, strideValue});
    }

    const json& accessors = optionalArray(root, "accessors", path);
    std::vector<std::size_t> accessorCounts;
    std::vector<std::size_t> accessorOffsets;
    accessorCounts.reserve(accessors.size());
    accessorOffsets.reserve(accessors.size());
    for (std::size_t index = 0; index < accessors.size(); ++index)
    {
        const json& accessor = accessors[index];
        const std::string context = "accessors[" + std::to_string(index) + "]";
        if (optionalMember(accessor, "sparse", path, context) != nullptr)
            fail(path, context + " sparse accessors are unsupported");
        const int viewIndex = indexValue(
            requiredMember(accessor, "bufferView", path, context), path,
            context + ".bufferView");
        if (static_cast<std::size_t>(viewIndex) >= views.size())
            fail(path, context + ".bufferView is out of range");
        const int componentType = indexValue(
            requiredMember(accessor, "componentType", path, context), path,
            context + ".componentType");
        const std::size_t width = rawComponentWidth(componentType, path, context);
        const std::size_t count = unsignedValue(
            requiredMember(accessor, "count", path, context), path,
            context + ".count");
        if (count == 0) fail(path, context + ".count must be positive");
        accessorCounts.push_back(count);
        const json& type = requiredMember(accessor, "type", path, context);
        if (!type.is_string()) fail(path, context + ".type must be a string");
        const RawAccessorShape shape = rawAccessorShape(
            type.get<std::string>(), path, context);
        const std::size_t elementSize = rawElementSize(width, shape, path, context);
        std::size_t offsetValue = 0;
        if (const json* offset = optionalMember(accessor, "byteOffset", path, context))
            offsetValue = unsignedValue(*offset, path, context + ".byteOffset");
        const RawView& view = rawViews[static_cast<std::size_t>(viewIndex)];
        std::size_t absoluteOffset = 0;
        if (!checkedAdd(view.byteOffset, offsetValue, &absoluteOffset))
            fail(path, context + " absolute byte offset overflows");
        if (offsetValue % width != 0 || absoluteOffset % width != 0)
            fail(path, context + " byte offset is misaligned");
        if (shape.matrix && width < 4
            && (offsetValue % 4 != 0 || absoluteOffset % 4 != 0))
            fail(path, context + " matrix byte offset must be four-byte aligned");
        accessorOffsets.push_back(offsetValue);
        const std::size_t stride = view.byteStride == 0
            ? elementSize : view.byteStride;
        if (stride < elementSize || stride % width != 0)
            fail(path, context + " byteStride is invalid for its element type");
        std::size_t repeated = 0;
        std::size_t relativeEnd = 0;
        if (!checkedMultiply(count - 1, stride, &repeated)
            || !checkedAdd(offsetValue, repeated, &relativeEnd)
            || !checkedAdd(relativeEnd, elementSize, &relativeEnd)
            || relativeEnd > view.byteLength)
            fail(path, context + " range exceeds its bufferView");
        if (const json* normalized = optionalMember(accessor, "normalized", path, context);
            normalized != nullptr)
        {
            if (!normalized->is_boolean())
                fail(path, context + ".normalized must be boolean");
            if (normalized->get<bool>()
                && componentType != TINYGLTF_COMPONENT_TYPE_BYTE
                && componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
                && componentType != TINYGLTF_COMPONENT_TYPE_SHORT
                && componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                fail(path, context + ".normalized is invalid for this component type");
        }
    }

    const json& nativeMaterials = optionalArray(root, "materials", path);
    const json& meshes = optionalArray(root, "meshes", path);
    for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
    {
        const json& mesh = meshes[meshIndex];
        const std::string meshContext = "meshes[" + std::to_string(meshIndex) + "]";
        const json& primitives = requiredMember(mesh, "primitives", path, meshContext);
        if (!primitives.is_array()) fail(path, meshContext + ".primitives must be an array");
        if (const json* weights = optionalMember(mesh, "weights", path, meshContext);
            weights != nullptr && (!weights->is_array() || !weights->empty()))
            fail(path, meshContext + " morph weights are unsupported");
        for (std::size_t primitiveIndex = 0; primitiveIndex < primitives.size();
             ++primitiveIndex)
        {
            const json& primitive = primitives[primitiveIndex];
            const std::string context = meshContext + ".primitives["
                + std::to_string(primitiveIndex) + "]";
            const json& attributes = requiredMember(primitive, "attributes", path,
                                                    context);
            if (!attributes.is_object()) fail(path, context + ".attributes must be an object");
            if (!attributes.contains("POSITION"))
                fail(path, context + " is missing the POSITION attribute");
            const int positionAccessor = indexValue(attributes.at("POSITION"), path,
                                                     context + ".attributes.POSITION");
            if (static_cast<std::size_t>(positionAccessor) >= accessors.size())
                fail(path, context + ".attributes.POSITION is out of range");
            const std::size_t vertexCount = accessorCounts[
                static_cast<std::size_t>(positionAccessor)];
            for (const auto& entry : attributes.items())
            {
                const int accessorIndex = indexValue(entry.value(), path,
                    context + ".attributes." + entry.key());
                if (static_cast<std::size_t>(accessorIndex) >= accessors.size())
                    fail(path, context + ".attributes." + entry.key()
                        + " is out of range");
                if (accessorCounts[static_cast<std::size_t>(accessorIndex)]
                    != vertexCount)
                    fail(path, context + ".attributes." + entry.key()
                        + " count does not match POSITION");
                if (accessorOffsets[static_cast<std::size_t>(accessorIndex)] % 4 != 0)
                    fail(path, context + ".attributes." + entry.key()
                        + " must be four-byte aligned");
                if (entry.key().rfind("JOINTS_", 0) == 0
                    || entry.key().rfind("WEIGHTS_", 0) == 0)
                    fail(path, context + " skinning attributes are unsupported");
            }
            if (const json* indices = optionalMember(primitive, "indices", path, context))
            {
                const int accessorIndex = indexValue(*indices, path,
                                                     context + ".indices");
                if (static_cast<std::size_t>(accessorIndex) >= accessors.size())
                    fail(path, context + ".indices is out of range");
                if (accessorCounts[static_cast<std::size_t>(accessorIndex)] % 3 != 0)
                    fail(path, context + " triangle index count must be a multiple of three");
            }
            else if (vertexCount % 3 != 0)
                fail(path, context + " unindexed POSITION count must be a multiple of three");
            if (const json* material = optionalMember(primitive, "material", path, context))
            {
                const int materialIndex = indexValue(*material, path,
                                                     context + ".material");
                if (static_cast<std::size_t>(materialIndex)
                    >= nativeMaterials.size())
                    fail(path, context + ".material is out of range");
            }
            if (const json* mode = optionalMember(primitive, "mode", path, context))
            {
                if (indexValue(*mode, path, context + ".mode")
                    != TINYGLTF_MODE_TRIANGLES)
                    fail(path, context + " uses an unsupported non-triangle mode");
            }
            if (const json* targets = optionalMember(primitive, "targets", path, context);
                targets != nullptr && (!targets->is_array() || !targets->empty()))
                fail(path, context + " morph targets are unsupported");
        }
    }

    const json& nodes = optionalArray(root, "nodes", path);
    for (std::size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex)
    {
        const json& node = nodes[nodeIndex];
        const std::string context = "nodes[" + std::to_string(nodeIndex) + "]";
        const json* matrix = optionalMember(node, "matrix", path, context);
        const bool hasTrs = optionalMember(node, "translation", path, context)
            || optionalMember(node, "rotation", path, context)
            || optionalMember(node, "scale", path, context);
        if (matrix != nullptr && hasTrs)
            fail(path, context + " must not combine matrix with TRS properties");
        validateRawVector(node, "matrix", 16, path, context);
        validateRawVector(node, "translation", 3, path, context);
        validateRawVector(node, "rotation", 4, path, context);
        validateRawVector(node, "scale", 3, path, context);
        if (const json* mesh = optionalMember(node, "mesh", path, context))
        {
            const int meshIndex = indexValue(*mesh, path, context + ".mesh");
            if (static_cast<std::size_t>(meshIndex) >= meshes.size())
                fail(path, context + ".mesh is out of range");
        }
        if (const json* skin = optionalMember(node, "skin", path, context))
        {
            (void)indexValue(*skin, path, context + ".skin");
            fail(path, context + " skinning is unsupported");
        }
        if (const json* weights = optionalMember(node, "weights", path, context);
            weights != nullptr && (!weights->is_array() || !weights->empty()))
            fail(path, context + " morph weights are unsupported");
        if (const json* children = optionalMember(node, "children", path, context))
        {
            if (!children->is_array()) fail(path, context + ".children must be an array");
            for (std::size_t child = 0; child < children->size(); ++child)
            {
                const int childIndex = indexValue((*children)[child], path,
                    context + ".children[" + std::to_string(child) + "]");
                if (static_cast<std::size_t>(childIndex) >= nodes.size())
                    fail(path, context + ".children[" + std::to_string(child)
                        + "] is out of range");
            }
        }
    }

    const json& scenes = optionalArray(root, "scenes", path);
    for (std::size_t sceneIndex = 0; sceneIndex < scenes.size(); ++sceneIndex)
    {
        const json& scene = scenes[sceneIndex];
        const std::string context = "scenes[" + std::to_string(sceneIndex) + "]";
        if (const json* roots = optionalMember(scene, "nodes", path, context))
        {
            if (!roots->is_array()) fail(path, context + ".nodes must be an array");
            for (std::size_t rootIndex = 0; rootIndex < roots->size(); ++rootIndex)
            {
                const int nodeIndex = indexValue((*roots)[rootIndex], path,
                    context + ".nodes[" + std::to_string(rootIndex) + "]");
                if (static_cast<std::size_t>(nodeIndex) >= nodes.size())
                    fail(path, context + ".nodes[" + std::to_string(rootIndex)
                        + "] is out of range");
            }
        }
    }
    if (const json* defaultScene = optionalMember(root, "scene", path, "root"))
    {
        const int sceneIndex = indexValue(*defaultScene, path, "scene");
        if (static_cast<std::size_t>(sceneIndex) >= scenes.size())
            fail(path, "default scene index is out of range");
    }
}

void requireParsedCounts(const tinygltf::Model& model, const json& root,
                         const fs::path& path)
{
    struct CountCheck
    {
        const char* name;
        std::size_t parsed;
    };
    const CountCheck checks[] = {
        {"buffers", model.buffers.size()},
        {"bufferViews", model.bufferViews.size()},
        {"accessors", model.accessors.size()},
        {"meshes", model.meshes.size()},
        {"nodes", model.nodes.size()},
        {"scenes", model.scenes.size()},
        {"images", model.images.size()},
    };
    for (const CountCheck& check : checks)
    {
        const std::size_t raw = root.contains(check.name)
            ? root.at(check.name).size() : 0;
        if (raw != check.parsed)
            fail(path, std::string("parser discarded malformed ") + check.name);
    }
    const json& rawMeshes = root.contains("meshes") ? root.at("meshes")
                                                     : json::array();
    for (std::size_t index = 0; index < model.meshes.size(); ++index)
    {
        if (rawMeshes[index].at("primitives").size()
            != model.meshes[index].primitives.size())
            fail(path, "parser discarded a malformed mesh primitive");
    }
}

glm::dmat4 nodeLocalMatrix(const tinygltf::Node& node, std::size_t index,
                           const fs::path& path)
{
    const std::string context = "node[" + std::to_string(index) + "]";
    glm::dmat4 result(1.0);
    if (!node.matrix.empty())
    {
        if (node.matrix.size() != 16)
            fail(path, context + " matrix must have sixteen components");
        for (std::size_t component = 0; component < 16; ++component)
        {
            if (!finite(node.matrix[component])
                || std::fabs(node.matrix[component])
                    > static_cast<double>(std::numeric_limits<float>::max()))
                fail(path, context + " matrix contains an invalid component");
            result[component / 4][component % 4] = node.matrix[component];
        }
        if (result[0][3] != 0.0 || result[1][3] != 0.0
            || result[2][3] != 0.0 || result[3][3] != 1.0)
            fail(path, context + " matrix must be affine");
    }
    else
    {
        glm::dvec3 translation(0.0);
        glm::dvec3 scale(1.0);
        glm::dquat rotation(1.0, 0.0, 0.0, 0.0);
        if (!node.translation.empty())
        {
            if (node.translation.size() != 3)
                fail(path, context + " translation must have three components");
            for (int axis = 0; axis < 3; ++axis)
            {
                if (!finite(node.translation[axis])
                    || std::fabs(node.translation[axis])
                        > static_cast<double>(std::numeric_limits<float>::max()))
                    fail(path, context + " translation contains an invalid component");
                translation[axis] = node.translation[axis];
            }
        }
        if (!node.scale.empty())
        {
            if (node.scale.size() != 3)
                fail(path, context + " scale must have three components");
            for (int axis = 0; axis < 3; ++axis)
            {
                if (!finite(node.scale[axis]) || node.scale[axis] == 0.0
                    || std::fabs(node.scale[axis])
                        > static_cast<double>(std::numeric_limits<float>::max()))
                    fail(path, context + " scale must be finite and nonsingular");
                scale[axis] = node.scale[axis];
            }
        }
        if (!node.rotation.empty())
        {
            if (node.rotation.size() != 4)
                fail(path, context + " rotation must have four components");
            for (double value : node.rotation)
            {
                if (!finite(value)
                    || std::fabs(value)
                        > static_cast<double>(std::numeric_limits<float>::max()))
                    fail(path, context + " quaternion contains an invalid component");
            }
            const double normSquared = node.rotation[0] * node.rotation[0]
                + node.rotation[1] * node.rotation[1]
                + node.rotation[2] * node.rotation[2]
                + node.rotation[3] * node.rotation[3];
            if (!finite(normSquared) || std::fabs(normSquared - 1.0) > 1.0e-6)
                fail(path, context + " quaternion must have unit norm");
            rotation = glm::dquat(node.rotation[3], node.rotation[0],
                                  node.rotation[1], node.rotation[2]);
        }
        result = glm::translate(glm::dmat4(1.0), translation)
            * glm::mat4_cast(rotation) * glm::scale(glm::dmat4(1.0), scale);
    }
    const double determinant = glm::determinant(glm::dmat3(result));
    if (!finite(determinant) || determinant == 0.0)
        fail(path, context + " transform is singular or non-finite");
    return result;
}

glm::mat4 floatMatrix(const glm::dmat4& matrix, const fs::path& path,
                      const std::string& context)
{
    glm::mat4 result(1.0f);
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            const double value = matrix[column][row];
            if (!finite(value)
                || value < -static_cast<double>(std::numeric_limits<float>::max())
                || value > static_cast<double>(std::numeric_limits<float>::max()))
                fail(path, context + " transform exceeds binary32 range");
            result[column][row] = static_cast<float>(value);
        }
    }
    const float determinant = glm::determinant(glm::mat3(result));
    if (!finite(result) || !finite(determinant) || determinant == 0.0f)
        fail(path, context + " transform is singular after binary32 conversion");
    return result;
}

struct AccessorBytes
{
    const unsigned char* bytes = nullptr;
    std::size_t count = 0;
    std::size_t stride = 0;
    std::size_t elementSize = 0;
    int componentType = -1;
    int type = -1;
    bool normalized = false;
};

std::size_t componentWidth(int componentType, const fs::path& path,
                           const std::string& context)
{
    switch (componentType)
    {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return 1;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT: return 4;
        default: fail(path, context + " uses an unsupported component type");
    }
}

std::size_t componentCount(int type, const fs::path& path,
                           const std::string& context)
{
    switch (type)
    {
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2: return 2;
        case TINYGLTF_TYPE_VEC3: return 3;
        default: fail(path, context + " uses an unsupported accessor shape");
    }
}

AccessorBytes accessorBytes(const tinygltf::Model& model, int accessorIndex,
                            bool indexAccessor, const fs::path& path,
                            const std::string& context)
{
    if (accessorIndex < 0
        || static_cast<std::size_t>(accessorIndex) >= model.accessors.size())
        fail(path, context + " accessor index is out of range");
    const tinygltf::Accessor& accessor = model.accessors[
        static_cast<std::size_t>(accessorIndex)];
    if (accessor.sparse.isSparse || accessor.sparse.count != 0
        || accessor.sparse.indices.bufferView != -1
        || accessor.sparse.values.bufferView != -1)
        fail(path, context + " sparse accessors are unsupported");
    if (accessor.bufferView < 0
        || static_cast<std::size_t>(accessor.bufferView) >= model.bufferViews.size())
        fail(path, context + " bufferView index is out of range");
    if (accessor.count == 0)
        fail(path, context + " accessor count must be positive");
    if (accessor.count > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
        fail(path, context + " accessor count exceeds renderer limits");

    const tinygltf::BufferView& view = model.bufferViews[
        static_cast<std::size_t>(accessor.bufferView)];
    if (view.buffer < 0
        || static_cast<std::size_t>(view.buffer) >= model.buffers.size())
        fail(path, context + " buffer index is out of range");
    const tinygltf::Buffer& buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
    if (view.byteLength == 0) fail(path, context + " bufferView is empty");

    std::size_t viewEnd = 0;
    if (!checkedAdd(view.byteOffset, view.byteLength, &viewEnd)
        || viewEnd > buffer.data.size())
        fail(path, context + " bufferView range exceeds its buffer");
    const std::size_t width = componentWidth(accessor.componentType, path, context);
    const std::size_t components = componentCount(accessor.type, path, context);
    std::size_t elementSize = 0;
    if (!checkedMultiply(width, components, &elementSize))
        fail(path, context + " accessor element size overflows");
    std::size_t absoluteStart = 0;
    if (!checkedAdd(view.byteOffset, accessor.byteOffset, &absoluteStart))
        fail(path, context + " accessor offset overflows");
    if (accessor.byteOffset % width != 0 || absoluteStart % width != 0)
        fail(path, context + " accessor offset is misaligned");

    std::size_t stride = elementSize;
    if (view.byteStride != 0)
    {
        if (indexAccessor)
            fail(path, context + " index accessors must not be strided");
        if (view.byteStride < elementSize || view.byteStride < 4
            || view.byteStride > 252 || view.byteStride % 4 != 0
            || view.byteStride % width != 0)
            fail(path, context + " byteStride is invalid for the accessor");
        stride = view.byteStride;
    }
    std::size_t repeated = 0;
    if (!checkedMultiply(accessor.count - 1, stride, &repeated))
        fail(path, context + " accessor count and stride overflow");
    std::size_t relativeEnd = 0;
    if (!checkedAdd(accessor.byteOffset, repeated, &relativeEnd)
        || !checkedAdd(relativeEnd, elementSize, &relativeEnd)
        || relativeEnd > view.byteLength)
        fail(path, context + " accessor range exceeds its bufferView");
    if (absoluteStart > buffer.data.size())
        fail(path, context + " accessor offset overflows");

    AccessorBytes result;
    result.bytes = buffer.data.data() + absoluteStart;
    result.count = accessor.count;
    result.stride = stride;
    result.elementSize = elementSize;
    result.componentType = accessor.componentType;
    result.type = accessor.type;
    result.normalized = accessor.normalized;
    return result;
}

std::vector<glm::vec3> decodeFloatVec3(const tinygltf::Model& model,
                                       int accessorIndex,
                                       const fs::path& path,
                                       const std::string& context,
                                       bool normals)
{
    const AccessorBytes accessor = accessorBytes(model, accessorIndex, false,
                                                 path, context);
    if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT
        || accessor.type != TINYGLTF_TYPE_VEC3 || accessor.normalized)
        fail(path, context + " must be non-normalized float VEC3");
    std::vector<glm::vec3> result(accessor.count);
    for (std::size_t index = 0; index < accessor.count; ++index)
    {
        const unsigned char* element = accessor.bytes + index * accessor.stride;
        glm::vec3 value(readFloat32(element), readFloat32(element + 4),
                        readFloat32(element + 8));
        const float lengthSquared = glm::dot(value, value);
        if (!finite(value) || (normals && (!finite(lengthSquared)
                                           || !(lengthSquared > 0.0f))))
            fail(path, context + " contains a non-finite or zero value");
        result[index] = value;
    }
    return result;
}

std::vector<glm::vec2> decodeTexcoords(const tinygltf::Model& model,
                                       int accessorIndex,
                                       const fs::path& path,
                                       const std::string& context)
{
    const AccessorBytes accessor = accessorBytes(model, accessorIndex, false,
                                                 path, context);
    if (accessor.type != TINYGLTF_TYPE_VEC2)
        fail(path, context + " must be VEC2");
    const bool floats = accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
        && !accessor.normalized;
    const bool bytes = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
        && accessor.normalized;
    const bool shorts = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
        && accessor.normalized;
    if (!floats && !bytes && !shorts)
        fail(path, context + " must be float or normalized unsigned VEC2");

    std::vector<glm::vec2> result(accessor.count);
    for (std::size_t index = 0; index < accessor.count; ++index)
    {
        const unsigned char* element = accessor.bytes + index * accessor.stride;
        glm::vec2 value;
        if (floats)
            value = glm::vec2(readFloat32(element), readFloat32(element + 4));
        else if (bytes)
            value = glm::vec2(static_cast<float>(element[0]) / 255.0f,
                              static_cast<float>(element[1]) / 255.0f);
        else
            value = glm::vec2(static_cast<float>(readU16(element)) / 65535.0f,
                              static_cast<float>(readU16(element + 2)) / 65535.0f);
        if (!finite(value)) fail(path, context + " contains a non-finite value");
        result[index] = glm::vec2(value.x, 1.0f - value.y);
    }
    return result;
}

std::vector<std::uint32_t> decodeIndices(const tinygltf::Model& model,
                                         int accessorIndex,
                                         const fs::path& path,
                                         const std::string& context)
{
    const AccessorBytes accessor = accessorBytes(model, accessorIndex, true,
                                                 path, context);
    if (accessor.type != TINYGLTF_TYPE_SCALAR || accessor.normalized
        || (accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
            && accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
            && accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT))
        fail(path, context + " must be an unsigned non-normalized SCALAR");
    std::vector<std::uint32_t> result(accessor.count);
    for (std::size_t index = 0; index < accessor.count; ++index)
    {
        const unsigned char* element = accessor.bytes + index * accessor.stride;
        if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
            result[index] = element[0];
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
            result[index] = readU16(element);
        else
            result[index] = readU32(element);
    }
    return result;
}

glm::vec3 transformPosition(const glm::mat4& transform,
                            const glm::vec3& position)
{
    return glm::vec3(transform * glm::vec4(position, 1.0f));
}

glm::vec3 transformNormal(const glm::mat4& inverseTranspose,
                          const glm::vec3& normal,
                          const fs::path& path,
                          const std::string& context)
{
    const glm::vec3 transformed(inverseTranspose * glm::vec4(normal, 0.0f));
    const float lengthSquared = glm::dot(transformed, transformed);
    if (!finite(transformed) || !finite(lengthSquared) || !(lengthSquared > 0.0f))
        fail(path, context + " produces an invalid transformed normal");
    return transformed / std::sqrt(lengthSquared);
}

void appendPrimitive(const tinygltf::Model& model,
                     const tinygltf::Primitive& primitive,
                     const Geom& sceneObject, const glm::dmat4& nodeWorld,
                     std::size_t nodeIndex, std::size_t primitiveIndex,
                     const fs::path& path, std::vector<Geom>* triangles)
{
    const std::string context = "node[" + std::to_string(nodeIndex)
        + "] primitive[" + std::to_string(primitiveIndex) + "]";
    const int mode = primitive.mode < 0 ? TINYGLTF_MODE_TRIANGLES : primitive.mode;
    if (mode != TINYGLTF_MODE_TRIANGLES)
        fail(path, context + " uses an unsupported non-triangle mode");
    if (!primitive.targets.empty()) fail(path, context + " morph targets are unsupported");
    for (const auto& attribute : primitive.attributes)
    {
        if (attribute.first.rfind("JOINTS_", 0) == 0
            || attribute.first.rfind("WEIGHTS_", 0) == 0)
            fail(path, context + " skinning attributes are unsupported");
    }
    const auto positionAttribute = primitive.attributes.find("POSITION");
    if (positionAttribute == primitive.attributes.end())
        fail(path, context + " is missing POSITION");

    const std::vector<glm::vec3> positions = decodeFloatVec3(model,
        positionAttribute->second, path, context + " POSITION", false);
    std::vector<glm::vec3> normals;
    if (const auto normal = primitive.attributes.find("NORMAL");
        normal != primitive.attributes.end())
    {
        normals = decodeFloatVec3(model, normal->second, path,
                                  context + " NORMAL", true);
        if (normals.size() != positions.size())
            fail(path, context + " NORMAL count does not match POSITION");
    }
    std::vector<glm::vec2> texcoords;
    if (const auto texcoord = primitive.attributes.find("TEXCOORD_0");
        texcoord != primitive.attributes.end())
    {
        texcoords = decodeTexcoords(model, texcoord->second, path,
                                    context + " TEXCOORD_0");
        if (texcoords.size() != positions.size())
            fail(path, context + " TEXCOORD_0 count does not match POSITION");
    }

    std::vector<std::uint32_t> indices;
    if (primitive.indices >= 0)
        indices = decodeIndices(model, primitive.indices, path, context + " indices");
    else
    {
        indices.resize(positions.size());
        for (std::size_t index = 0; index < positions.size(); ++index)
            indices[index] = static_cast<std::uint32_t>(index);
    }
    if (indices.empty() || indices.size() % 3 != 0)
        fail(path, context + " triangle index count must be a positive multiple of three");
    for (std::uint32_t index : indices)
    {
        if (index >= positions.size()) fail(path, context + " index is out of range");
    }
    const std::size_t triangleCount = indices.size() / 3;
    if (triangleCount > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || triangles->size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
            - triangleCount)
        fail(path, "imported triangle count exceeds renderer limits");

    const glm::dmat4 objectMatrix(sceneObject.transform);
    const glm::mat4 total = floatMatrix(objectMatrix * nodeWorld, path, context);
    const float determinant = glm::determinant(glm::mat3(total));
    const bool mirrored = determinant < 0.0f;
    const glm::mat4 inverse = glm::inverse(total);
    const glm::mat4 inverseTranspose = glm::transpose(inverse);
    if (!finite(inverse) || !finite(inverseTranspose))
        fail(path, context + " total transform has a non-finite inverse");

    triangles->reserve(triangles->size() + triangleCount);
    for (std::size_t triangleIndex = 0; triangleIndex < triangleCount;
         ++triangleIndex)
    {
        std::uint32_t corners[3] = {
            indices[3 * triangleIndex], indices[3 * triangleIndex + 1],
            indices[3 * triangleIndex + 2]};
        if (mirrored) std::swap(corners[1], corners[2]);

        Geom triangle = sceneObject;
        triangle.type = TRIANGLE;
        triangle.transform = total;
        triangle.inverseTransform = inverse;
        triangle.invTranspose = inverseTranspose;
        triangle.hasNormals = normals.empty() ? 0 : 1;
        triangle.hasUVs = texcoords.empty() ? 0 : 1;
        for (int corner = 0; corner < 3; ++corner)
        {
            triangle.vertices[corner] = transformPosition(total,
                positions[corners[corner]]);
            if (!finite(triangle.vertices[corner]))
                fail(path, context + " transform produces a non-finite vertex");
        }
        const glm::vec3 edge1 = triangle.vertices[1] - triangle.vertices[0];
        const glm::vec3 edge2 = triangle.vertices[2] - triangle.vertices[0];
        const glm::vec3 cross = glm::cross(edge1, edge2);
        const float crossLengthSquared = glm::dot(cross, cross);
        const float edgeProductSquared = glm::dot(edge1, edge1)
            * glm::dot(edge2, edge2);
        if (!finite(cross) || !finite(crossLengthSquared)
            || !finite(edgeProductSquared) || !(edgeProductSquared > 0.0f)
            || crossLengthSquared <= edgeProductSquared * 1.0e-14f)
            fail(path, context + " transform produces a degenerate triangle");
        const glm::vec3 flatNormal = cross / std::sqrt(crossLengthSquared);

        if (triangle.hasNormals)
        {
            for (int corner = 0; corner < 3; ++corner)
            {
                triangle.normals[corner] = transformNormal(inverseTranspose,
                    normals[corners[corner]], path, context);
                if (glm::dot(triangle.normals[corner], flatNormal) < 0.0f)
                    triangle.normals[corner] = -triangle.normals[corner];
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
                triangle.uvs[corner] = texcoords[corners[corner]];
        }
        else
        {
            triangle.uvs[0] = glm::vec2(0.0f, 0.0f);
            triangle.uvs[1] = glm::vec2(1.0f, 0.0f);
            triangle.uvs[2] = glm::vec2(0.0f, 1.0f);
        }
        triangles->push_back(triangle);
    }
}

std::vector<int> validateHierarchyAndRoots(const tinygltf::Model& model,
                                           const fs::path& path)
{
    if (model.nodes.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max()))
        fail(path, "node count exceeds renderer limits");
    std::vector<int> parentCounts(model.nodes.size(), 0);
    for (std::size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex)
    {
        const tinygltf::Node& node = model.nodes[nodeIndex];
        if (node.mesh < -1
            || (node.mesh >= 0
                && static_cast<std::size_t>(node.mesh) >= model.meshes.size()))
            fail(path, "node mesh index is out of range");
        if (node.skin != -1) fail(path, "skinned nodes are unsupported");
        if (!node.weights.empty()) fail(path, "node morph weights are unsupported");
        std::unordered_set<int> localChildren;
        for (int child : node.children)
        {
            if (child < 0 || static_cast<std::size_t>(child) >= model.nodes.size())
                fail(path, "node child index is out of range");
            if (!localChildren.insert(child).second)
                fail(path, "node contains a duplicate child index");
            if (++parentCounts[static_cast<std::size_t>(child)] > 1)
                fail(path, "node hierarchy assigns more than one parent");
        }
    }

    struct Frame { int node; std::size_t nextChild; std::size_t depth; };
    std::vector<unsigned char> color(model.nodes.size(), 0);
    std::vector<Frame> stack;
    for (std::size_t start = 0; start < model.nodes.size(); ++start)
    {
        if (color[start] != 0) continue;
        color[start] = 1;
        stack.push_back({static_cast<int>(start), 0, 1});
        while (!stack.empty())
        {
            Frame& frame = stack.back();
            if (frame.depth > kMaximumHierarchyDepth)
                fail(path, "node hierarchy exceeds maximum depth 4096");
            const auto& children = model.nodes[static_cast<std::size_t>(
                frame.node)].children;
            if (frame.nextChild == children.size())
            {
                color[static_cast<std::size_t>(frame.node)] = 2;
                stack.pop_back();
                continue;
            }
            const int child = children[frame.nextChild++];
            if (color[static_cast<std::size_t>(child)] == 1)
                fail(path, "node hierarchy contains a cycle");
            if (color[static_cast<std::size_t>(child)] == 0)
            {
                color[static_cast<std::size_t>(child)] = 1;
                stack.push_back({child, 0, frame.depth + 1});
            }
        }
    }

    std::vector<int> roots;
    if (!model.scenes.empty())
    {
        int sceneIndex = model.defaultScene;
        if (sceneIndex == -1) sceneIndex = 0;
        if (sceneIndex < 0
            || static_cast<std::size_t>(sceneIndex) >= model.scenes.size())
            fail(path, "default scene index is out of range");
        std::unordered_set<int> uniqueRoots;
        for (int root : model.scenes[static_cast<std::size_t>(sceneIndex)].nodes)
        {
            if (root < 0 || static_cast<std::size_t>(root) >= model.nodes.size())
                fail(path, "scene root node index is out of range");
            if (parentCounts[static_cast<std::size_t>(root)] != 0)
                fail(path, "scene root node also has a parent");
            if (!uniqueRoots.insert(root).second)
                fail(path, "scene contains a duplicate root node");
            roots.push_back(root);
        }
    }
    else
    {
        if (model.defaultScene != -1)
            fail(path, "default scene index is invalid when scenes are absent");
        // A glTF without scene definitions is imported from every parentless
        // node, in node-index order. This deterministically covers all coherent
        // hierarchy roots without treating disconnected child nodes as roots.
        for (std::size_t node = 0; node < model.nodes.size(); ++node)
            if (parentCounts[node] == 0) roots.push_back(static_cast<int>(node));
    }
    return roots;
}

void importSelectedScene(const tinygltf::Model& model, const Geom& sceneObject,
                         const fs::path& path, std::vector<Geom>* triangles)
{
    for (const tinygltf::Mesh& mesh : model.meshes)
    {
        if (!mesh.weights.empty()) fail(path, "mesh morph weights are unsupported");
    }
    if (!model.extensionsRequired.empty())
        fail(path, "required extensions are unsupported");
    if (!model.animations.empty()) fail(path, "animations are unsupported");
    if (!model.skins.empty()) fail(path, "skins are unsupported");

    const std::vector<int> roots = validateHierarchyAndRoots(model, path);
    std::vector<glm::dmat4> localMatrices;
    localMatrices.reserve(model.nodes.size());
    for (std::size_t index = 0; index < model.nodes.size(); ++index)
        localMatrices.push_back(nodeLocalMatrix(model.nodes[index], index, path));

    struct Task { int node; glm::dmat4 parent; std::size_t depth; };
    std::vector<Task> stack;
    for (auto root = roots.rbegin(); root != roots.rend(); ++root)
        stack.push_back({*root, glm::dmat4(1.0), 1});
    std::vector<unsigned char> visited(model.nodes.size(), 0);
    while (!stack.empty())
    {
        Task task = stack.back();
        stack.pop_back();
        if (task.depth > kMaximumHierarchyDepth)
            fail(path, "node hierarchy exceeds maximum depth 4096");
        if (visited[static_cast<std::size_t>(task.node)] != 0)
            fail(path, "selected scene visits a node more than once");
        visited[static_cast<std::size_t>(task.node)] = 1;
        const tinygltf::Node& node = model.nodes[static_cast<std::size_t>(task.node)];
        const glm::dmat4 world = task.parent
            * localMatrices[static_cast<std::size_t>(task.node)];
        (void)floatMatrix(glm::dmat4(sceneObject.transform) * world, path,
                          "node[" + std::to_string(task.node) + "]");
        if (node.mesh >= 0)
        {
            const tinygltf::Mesh& mesh = model.meshes[
                static_cast<std::size_t>(node.mesh)];
            for (std::size_t primitive = 0; primitive < mesh.primitives.size();
                 ++primitive)
                appendPrimitive(model, mesh.primitives[primitive], sceneObject,
                    world, static_cast<std::size_t>(task.node), primitive, path,
                    triangles);
        }
        for (auto child = node.children.rbegin(); child != node.children.rend();
             ++child)
            stack.push_back({*child, world, task.depth + 1});
    }
    if (triangles->empty())
        fail(path, "selected scene contains no triangle geometry");
}

bool ignoreImage(tinygltf::Image*, int, std::string*, std::string*, int, int,
                 const unsigned char* bytes, int size, void*)
{
    return bytes != nullptr && size > 0;
}

} // namespace

LoadResult load(const fs::path& canonicalModelPath, const Geom& sceneObject)
{
    ParsedDocument document = parseDocument(canonicalModelPath);
    LoadResult result;
    validateRawModel(document, canonicalModelPath, &result.externalAssets);

    tinygltf::TinyGLTF parser;
    parser.SetParseStrictness(tinygltf::ParseStrictness::Strict);
    parser.SetImageLoader(ignoreImage, nullptr);
    std::string uriError;
    tinygltf::URICallbacks callbacks;
    callbacks.decode = strictUriDecode;
    callbacks.user_data = &uriError;
    std::string callbackError;
    if (!parser.SetURICallbacks(std::move(callbacks), &callbackError))
        fail(canonicalModelPath, "could not configure strict URI decoding: "
            + callbackError);

    tinygltf::Model model;
    std::string parserError;
    std::string parserWarning;
    const std::string baseDirectory = canonicalModelPath.parent_path().string();
    bool loaded = false;
    if (document.binary)
    {
        loaded = parser.LoadBinaryFromMemory(&model, &parserError, &parserWarning,
            document.fileBytes.data(),
            static_cast<unsigned int>(document.fileBytes.size()), baseDirectory);
    }
    else
    {
        loaded = parser.LoadASCIIFromString(&model, &parserError, &parserWarning,
            document.jsonText.data(),
            static_cast<unsigned int>(document.jsonText.size()), baseDirectory);
    }
    if (!loaded)
    {
        std::string detail = !uriError.empty() ? uriError : parserError;
        if (detail.empty()) detail = parserWarning;
        fail(canonicalModelPath, "parser rejected the model"
            + (detail.empty() ? std::string() : ": " + detail));
    }
    requireParsedCounts(model, document.root, canonicalModelPath);
    importSelectedScene(model, sceneObject, canonicalModelPath,
                        &result.triangles);
    return result;
}

} // namespace gltf
