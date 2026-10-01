#include "checkpoint.h"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "utilities.h"

#include "json.hpp"
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr std::uint32_t kSumFileVersion = 1;

struct Options
{
    fs::path scenePath;
    std::optional<std::uint64_t> samples;
    std::optional<std::uint64_t> seed;
    std::optional<fs::path> outputPrefix;
    std::optional<std::pair<int, int>> resolution;
    std::optional<int> depth;
    std::optional<bool> aa;
    std::optional<bool> nee;
    std::optional<bool> bvh;
    std::optional<bool> compact;
    std::optional<bool> sort;
    std::optional<bool> wavefront;
    std::optional<bool> rr;
    std::optional<int> rrStart;
    std::optional<RendererSampler> sampler;
    std::optional<bool> motion;
    std::optional<RendererDebugMode> debug;
    std::optional<fs::path> checkpointPath;
    std::optional<std::uint64_t> checkpointEvery;
    std::optional<fs::path> resumePath;
    std::optional<std::uint64_t> timeBudgetMs;
    bool benchmark = false;
    bool validate = false;
    bool help = false;
};

struct ResolvedRun
{
    std::uint64_t targetSamples = 0;
    RendererConfig renderer;
    int width = 0;
    int height = 0;
    fs::path outputPrefix;
};

class RendererLifetime
{
public:
    void initialize(Scene& scene)
    {
        initialized_ = true;
        pathtraceInit(&scene);
    }

    void shutdown()
    {
        if (initialized_)
        {
            pathtraceFree();
            initialized_ = false;
        }
    }

    ~RendererLifetime() noexcept
    {
        if (initialized_)
        {
            try
            {
                shutdown();
            }
            catch (...)
            {
                // Preserve the original exception during stack unwinding.
            }
        }
    }

    RendererLifetime(const RendererLifetime&) = delete;
    RendererLifetime& operator=(const RendererLifetime&) = delete;
    RendererLifetime() = default;

private:
    bool initialized_ = false;
};

class CoutRedirect
{
public:
    explicit CoutRedirect(std::ostream& destination)
        : previous_(std::cout.rdbuf(destination.rdbuf()))
    {
    }

    ~CoutRedirect()
    {
        std::cout.rdbuf(previous_);
    }

    CoutRedirect(const CoutRedirect&) = delete;
    CoutRedirect& operator=(const CoutRedirect&) = delete;

private:
    std::streambuf* previous_;
};

std::string usage(std::string_view program)
{
    std::ostringstream text;
    text
        << "Usage: " << program << " SCENE.json [options]\n"
        << "\n"
        << "Required for normal rendering:\n"
        << "  --samples N              Total completed sample target (resume-safe)\n"
        << "  --output PREFIX          Unique output prefix; writes .png, .sum.bin, .json\n"
        << "\n"
        << "Renderer options:\n"
        << "  --seed N                 Unsigned 64-bit deterministic seed\n"
        << "  --resolution WIDTHxHEIGHT\n"
        << "  --depth N                Maximum surface intersections\n"
        << "  --aa on|off\n"
        << "  --nee on|off\n"
        << "  --bvh on|off\n"
        << "  --compact on|off\n"
        << "  --sort on|off\n"
        << "  --wavefront on|off\n"
        << "  --rr on|off\n"
        << "  --rr-start N             Completed scattering events before roulette (1..4096)\n"
        << "  --sampler counter|halton Sampling algorithm (default: counter)\n"
        << "  --motion on|off          Rigid-translation motion blur (default: on)\n"
        << "  --debug normals|uv|depth  Deterministic first-hit semantic view\n"
        << "\n"
        << "Checkpoint and diagnostics:\n"
        << "  --checkpoint PATH        Atomically save final checkpoint\n"
        << "  --checkpoint-every N     Also save every N completed samples\n"
        << "  --resume PATH            Resume a compatible checkpoint\n"
        << "  --time-budget-ms N       Fresh-render diagnostic budget; --samples is the ceiling\n"
        << "  --benchmark              Write PREFIX.benchmark.csv\n"
        << "  --validate               Run GPU equivalence/edge validation and exit\n"
        << "  --help\n";
    return text.str();
}

template<typename UInt>
UInt parseUnsigned(std::string_view text, std::string_view option)
{
    if (text.empty() || text.front() == '+' || text.front() == '-')
    {
        throw std::invalid_argument(std::string(option) + " expects an unsigned integer");
    }

    UInt result = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto conversion = std::from_chars(begin, end, result, 10);
    if (conversion.ec == std::errc::result_out_of_range)
    {
        throw std::out_of_range(std::string(option) + " is out of range");
    }
    if (conversion.ec != std::errc() || conversion.ptr != end)
    {
        throw std::invalid_argument(std::string(option) + " expects an unsigned integer");
    }
    return result;
}

bool parseSwitch(std::string_view text, std::string_view option)
{
    if (text == "on")
    {
        return true;
    }
    if (text == "off")
    {
        return false;
    }
    throw std::invalid_argument(std::string(option) + " expects exactly 'on' or 'off'");
}

RendererDebugMode parseDebugMode(std::string_view text)
{
    if (text == "normals") return RendererDebugMode::Normals;
    if (text == "uv") return RendererDebugMode::Uv;
    if (text == "depth") return RendererDebugMode::Depth;
    throw std::invalid_argument(
        "--debug expects exactly 'normals', 'uv', or 'depth'");
}

RendererSampler parseSampler(std::string_view text)
{
    if (text == "counter") return RendererSampler::Counter;
    if (text == "halton") return RendererSampler::Halton;
    throw std::invalid_argument(
        "--sampler expects exactly 'counter' or 'halton'");
}

const char* samplerName(RendererSampler sampler)
{
    switch (sampler)
    {
        case RendererSampler::Counter: return "counter";
        case RendererSampler::Halton: return "halton";
    }
    return "invalid";
}

const char* debugModeName(RendererDebugMode mode)
{
    switch (mode)
    {
        case RendererDebugMode::None: return "none";
        case RendererDebugMode::Normals: return "normals";
        case RendererDebugMode::Uv: return "uv";
        case RendererDebugMode::Depth: return "depth";
    }
    return "invalid";
}

std::pair<int, int> parseResolution(std::string_view text)
{
    const std::size_t separator = text.find_first_of("xX");
    if (separator == std::string_view::npos
        || separator == 0
        || separator + 1 == text.size()
        || text.find_first_of("xX", separator + 1) != std::string_view::npos)
    {
        throw std::invalid_argument("--resolution expects WIDTHxHEIGHT");
    }

    const std::uint64_t width = parseUnsigned<std::uint64_t>(
        text.substr(0, separator), "--resolution width");
    const std::uint64_t height = parseUnsigned<std::uint64_t>(
        text.substr(separator + 1), "--resolution height");
    if (width == 0 || height == 0
        || width > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        || height > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        || width > static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 3)
        || width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / height)
    {
        throw std::out_of_range("--resolution dimensions are zero or too large");
    }
    return {static_cast<int>(width), static_cast<int>(height)};
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    bool positionalOnly = false;

    const auto requireValue = [&](int& index, std::string_view option) -> std::string_view
    {
        if (index + 1 >= argc)
        {
            throw std::invalid_argument(std::string(option) + " requires a value");
        }
        ++index;
        return argv[index];
    };

    const auto rejectDuplicate = [](bool alreadySet, std::string_view option)
    {
        if (alreadySet)
        {
            throw std::invalid_argument(std::string(option) + " was specified more than once");
        }
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument(argv[i]);
        if (!positionalOnly && argument == "--")
        {
            positionalOnly = true;
            continue;
        }
        if (!positionalOnly && argument == "--help")
        {
            rejectDuplicate(options.help, argument);
            options.help = true;
        }
        else if (!positionalOnly && argument == "--samples")
        {
            rejectDuplicate(options.samples.has_value(), argument);
            const std::uint64_t value = parseUnsigned<std::uint64_t>(
                requireValue(i, argument), argument);
            if (value == 0 || value > 0x7fffffffffffffffull)
            {
                throw std::out_of_range("--samples must be in [1, 2^63-1]");
            }
            options.samples = value;
        }
        else if (!positionalOnly && argument == "--seed")
        {
            rejectDuplicate(options.seed.has_value(), argument);
            options.seed = parseUnsigned<std::uint64_t>(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--output")
        {
            rejectDuplicate(options.outputPrefix.has_value(), argument);
            const std::string value(requireValue(i, argument));
            if (value.empty())
            {
                throw std::invalid_argument("--output must not be empty");
            }
            options.outputPrefix = fs::path(value);
        }
        else if (!positionalOnly && argument == "--resolution")
        {
            rejectDuplicate(options.resolution.has_value(), argument);
            options.resolution = parseResolution(requireValue(i, argument));
        }
        else if (!positionalOnly && argument == "--depth")
        {
            rejectDuplicate(options.depth.has_value(), argument);
            const std::uint64_t value = parseUnsigned<std::uint64_t>(
                requireValue(i, argument), argument);
            if (value == 0 || value > 4096)
            {
                throw std::out_of_range("--depth must be in [1, 4096]");
            }
            options.depth = static_cast<int>(value);
        }
        else if (!positionalOnly && argument == "--aa")
        {
            rejectDuplicate(options.aa.has_value(), argument);
            options.aa = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--nee")
        {
            rejectDuplicate(options.nee.has_value(), argument);
            options.nee = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--bvh")
        {
            rejectDuplicate(options.bvh.has_value(), argument);
            options.bvh = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--compact")
        {
            rejectDuplicate(options.compact.has_value(), argument);
            options.compact = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--sort")
        {
            rejectDuplicate(options.sort.has_value(), argument);
            options.sort = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--wavefront")
        {
            rejectDuplicate(options.wavefront.has_value(), argument);
            options.wavefront = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--rr")
        {
            rejectDuplicate(options.rr.has_value(), argument);
            options.rr = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--rr-start")
        {
            rejectDuplicate(options.rrStart.has_value(), argument);
            const std::uint64_t value = parseUnsigned<std::uint64_t>(
                requireValue(i, argument), argument);
            if (value == 0 || value > 4096)
            {
                throw std::out_of_range("--rr-start must be in [1, 4096]");
            }
            options.rrStart = static_cast<int>(value);
        }
        else if (!positionalOnly && argument == "--sampler")
        {
            rejectDuplicate(options.sampler.has_value(), argument);
            options.sampler = parseSampler(requireValue(i, argument));
        }
        else if (!positionalOnly && argument == "--motion")
        {
            rejectDuplicate(options.motion.has_value(), argument);
            options.motion = parseSwitch(requireValue(i, argument), argument);
        }
        else if (!positionalOnly && argument == "--debug")
        {
            rejectDuplicate(options.debug.has_value(), argument);
            options.debug = parseDebugMode(requireValue(i, argument));
        }
        else if (!positionalOnly && argument == "--checkpoint")
        {
            rejectDuplicate(options.checkpointPath.has_value(), argument);
            const std::string value(requireValue(i, argument));
            if (value.empty())
            {
                throw std::invalid_argument("--checkpoint must not be empty");
            }
            options.checkpointPath = fs::path(value);
        }
        else if (!positionalOnly && argument == "--checkpoint-every")
        {
            rejectDuplicate(options.checkpointEvery.has_value(), argument);
            const std::uint64_t value = parseUnsigned<std::uint64_t>(
                requireValue(i, argument), argument);
            if (value == 0)
            {
                throw std::out_of_range("--checkpoint-every must be positive");
            }
            options.checkpointEvery = value;
        }
        else if (!positionalOnly && argument == "--resume")
        {
            rejectDuplicate(options.resumePath.has_value(), argument);
            const std::string value(requireValue(i, argument));
            if (value.empty())
            {
                throw std::invalid_argument("--resume must not be empty");
            }
            options.resumePath = fs::path(value);
        }
        else if (!positionalOnly && argument == "--time-budget-ms")
        {
            rejectDuplicate(options.timeBudgetMs.has_value(), argument);
            const std::uint64_t value = parseUnsigned<std::uint64_t>(
                requireValue(i, argument), argument);
            if (value == 0 || value > 0x7fffffffffffffffull)
            {
                throw std::out_of_range(
                    "--time-budget-ms must be in [1, 2^63-1]");
            }
            options.timeBudgetMs = value;
        }
        else if (!positionalOnly && argument == "--benchmark")
        {
            rejectDuplicate(options.benchmark, argument);
            options.benchmark = true;
        }
        else if (!positionalOnly && argument == "--validate")
        {
            rejectDuplicate(options.validate, argument);
            options.validate = true;
        }
        else if (!positionalOnly && argument.size() >= 2
                 && argument[0] == '-' && argument[1] == '-')
        {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
        else
        {
            if (!options.scenePath.empty())
            {
                throw std::invalid_argument("more than one scene path was supplied");
            }
            options.scenePath = fs::path(argument);
        }
    }

    if (options.help)
    {
        return options;
    }
    if (options.scenePath.empty())
    {
        throw std::invalid_argument("a scene JSON path is required");
    }
    if (options.timeBudgetMs && !options.samples)
    {
        throw std::invalid_argument(
            "--time-budget-ms requires an explicit --samples safety ceiling");
    }
    if (options.timeBudgetMs && (options.resumePath || options.checkpointPath
                                 || options.checkpointEvery))
    {
        throw std::invalid_argument(
            "--time-budget-ms cannot be combined with resume or checkpoint options");
    }
    if (options.timeBudgetMs && options.validate)
    {
        throw std::invalid_argument(
            "--time-budget-ms is only valid for a fresh render");
    }
    if (options.checkpointEvery && !options.checkpointPath)
    {
        throw std::invalid_argument("--checkpoint-every requires --checkpoint PATH");
    }
    if (!options.validate && !options.samples)
    {
        throw std::invalid_argument("--samples is required for normal rendering");
    }
    if (!options.validate && !options.outputPrefix)
    {
        throw std::invalid_argument("--output is required for normal rendering");
    }
    if (options.validate && (options.checkpointPath || options.checkpointEvery
                             || options.resumePath || options.benchmark))
    {
        throw std::invalid_argument(
            "--validate cannot be combined with checkpoint, resume, or benchmark options");
    }
    if (options.debug && (options.checkpointPath || options.checkpointEvery
                          || options.resumePath || options.benchmark
                          || options.validate))
    {
        throw std::invalid_argument(
            "--debug cannot be combined with checkpoint, resume, benchmark, or validation");
    }
    return options;
}

void requireRegularInput(const fs::path& path, std::string_view description)
{
    std::error_code error;
    if (!fs::exists(path, error) || error || !fs::is_regular_file(path, error) || error)
    {
        throw std::runtime_error(std::string(description) + " is not a readable regular file: "
                                 + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("cannot read " + std::string(description) + ": " + path.string());
    }
}

fs::path parentDirectory(const fs::path& path)
{
    return path.has_parent_path() ? path.parent_path() : fs::current_path();
}

void probeWritableSibling(const fs::path& path)
{
    std::error_code targetError;
    if (fs::exists(path, targetError)
        && (targetError || !fs::is_regular_file(path, targetError)))
    {
        throw std::runtime_error("output target is not a regular file: " + path.string());
    }
    if (targetError)
    {
        throw std::runtime_error("cannot inspect output target: " + path.string());
    }

    const fs::path directory = parentDirectory(path);
    std::error_code error;
    if (!fs::exists(directory, error) || error || !fs::is_directory(directory, error) || error)
    {
        throw std::runtime_error("output directory does not exist: " + directory.string());
    }

    fs::path probe = path;
    probe += ".cis565-write-test.tmp";
    if (fs::exists(probe, error) && !error)
    {
        throw std::runtime_error("refusing to overwrite stale write probe: " + probe.string());
    }

    {
        std::ofstream output(probe, std::ios::binary | std::ios::out);
        if (!output)
        {
            throw std::runtime_error("output location is not writable: " + directory.string());
        }
        output.put('\0');
        output.flush();
        if (!output)
        {
            throw std::runtime_error("failed write probe in: " + directory.string());
        }
    }

    if (!fs::remove(probe, error) || error)
    {
        throw std::runtime_error("failed to remove output write probe: " + probe.string());
    }
}

fs::path suffixed(const fs::path& prefix, std::string_view suffix)
{
    return fs::path(prefix.string() + std::string(suffix));
}

fs::path normalizedAbsolute(const fs::path& path)
{
    return fs::absolute(path).lexically_normal();
}

void validateUniqueOutputs(const fs::path& prefix,
                           bool benchmark,
                           const std::optional<fs::path>& checkpointPath)
{
    if (prefix.empty() || prefix.filename().empty())
    {
        throw std::invalid_argument("output prefix must include a filename");
    }

    std::vector<fs::path> outputs = {
        suffixed(prefix, ".png"),
        suffixed(prefix, ".sum.bin"),
        suffixed(prefix, ".json")};
    if (benchmark)
    {
        outputs.push_back(suffixed(prefix, ".benchmark.csv"));
    }

    if (checkpointPath)
    {
        const fs::path normalizedCheckpoint = normalizedAbsolute(*checkpointPath);
        for (const fs::path& output : outputs)
        {
            if (normalizedAbsolute(output) == normalizedCheckpoint)
            {
                throw std::invalid_argument(
                    "checkpoint path must not overlap a render output: "
                    + output.string());
            }
        }
    }

    std::error_code error;
    for (const fs::path& output : outputs)
    {
        const bool exists = fs::exists(output, error);
        if (error)
        {
            throw std::runtime_error("cannot inspect output path: " + output.string());
        }
        if (exists)
        {
            throw std::runtime_error("refusing to overwrite existing output: " + output.string());
        }
        error.clear();
    }
    probeWritableSibling(outputs.front());
}

void configureCameraResolution(Scene& scene, int width, int height)
{
    Camera& camera = scene.state.camera;
    if (camera.resolution.x == width && camera.resolution.y == height)
    {
        return;
    }
    camera.resolution = glm::ivec2(width, height);

    // Match the scene loader's float operation order for the same projection.
    constexpr float projectionPi = 3.14159265358979323846f;
    const float verticalScale = std::tan(camera.fov.y * projectionPi / 180.0f);
    const float horizontalScale = verticalScale
        * static_cast<float>(width) / static_cast<float>(height);
    if (!std::isfinite(verticalScale) || !std::isfinite(horizontalScale)
        || verticalScale < 0.0f || horizontalScale < 0.0f)
    {
        throw std::runtime_error("scene camera has an invalid field of view");
    }

    camera.fov.x = std::atan(horizontalScale) * 180.0f / projectionPi;
    camera.pixelLength = glm::vec2(
        2.0f * horizontalScale / static_cast<float>(width),
        2.0f * verticalScale / static_cast<float>(height));
    scene.state.image.assign(static_cast<std::size_t>(width)
                                 * static_cast<std::size_t>(height),
                             glm::vec3(0.0f));
}

checkpoint::RenderConfig checkpointConfig(const ResolvedRun& run)
{
    checkpoint::RenderConfig result;
    result.width = static_cast<std::uint32_t>(run.width);
    result.height = static_cast<std::uint32_t>(run.height);
    result.maxDepth = static_cast<std::uint32_t>(run.renderer.maxDepth);
    result.antialias = run.renderer.aa;
    result.nextEventEstimation = run.renderer.nee;
    result.useBvh = run.renderer.useBvh;
    result.compact = run.renderer.compact;
    result.materialSort = run.renderer.sort;
    result.wavefront = run.renderer.wavefront;
    result.russianRoulette = run.renderer.rr;
    result.rrStart = static_cast<std::uint32_t>(run.renderer.rrStart);
    result.motion = run.renderer.motion;
    return result;
}

checkpoint::CameraState checkpointCamera(const Camera& camera)
{
    checkpoint::CameraState result;
    result.position = camera.position;
    result.lookAt = camera.lookAt;
    result.view = camera.view;
    result.up = camera.up;
    result.right = camera.right;
    result.fov = camera.fov;
    result.pixelLength = camera.pixelLength;
    result.aperture = camera.aperture;
    result.focusDistance = camera.focusDistance;
    result.shutterOpen = camera.shutterOpen;
    result.shutterClose = camera.shutterClose;
    return result;
}

checkpoint::CompatibilityInfo checkpointCompatibility()
{
    const RendererCompatibilityInfo source = pathtraceCompatibilityInfo();
    checkpoint::CompatibilityInfo result;
    result.cudaRuntimeVersion = static_cast<std::uint32_t>(source.runtimeVersion);
    result.cudaDriverVersion = static_cast<std::uint32_t>(source.driverVersion);
    result.computeCapabilityMajor = static_cast<std::uint32_t>(source.computeMajor);
    result.computeCapabilityMinor = static_cast<std::uint32_t>(source.computeMinor);
    result.deviceIdentifier = source.deviceName;
    result.sourceRevision =
        "authorized-starter:8695a5af18ae6fc130d1d38713be82bbbff7f38f";
    result.buildIdentifier = source.sourceBuild;
    return result;
}

void addMatrix(checkpoint::FingerprintBuilder& builder, const glm::mat4& matrix)
{
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            builder.addFloat(matrix[column][row]);
        }
    }
}

checkpoint::Fingerprints fingerprintsFor(const fs::path& scenePath,
                                          const Scene& scene,
                                          const checkpoint::RenderConfig& config,
                                          std::uint32_t samplerId)
{
    checkpoint::Fingerprints result;
    checkpoint::Status status = checkpoint::fingerprintFile(scenePath, result.sceneFile);
    if (!status)
    {
        throw std::runtime_error("failed to fingerprint scene: " + status.message);
    }

    checkpoint::FingerprintBuilder assets;
    assets.addU64(static_cast<std::uint64_t>(scene.assets.size()));
    for (const std::string& assetName : scene.assets)
    {
        fs::path assetPath(assetName);
        if (assetPath.is_relative() && !fs::exists(assetPath))
        {
            assetPath = scenePath.parent_path() / assetPath;
        }
        std::uint64_t content = 0;
        status = checkpoint::fingerprintFile(assetPath, content);
        if (!status)
        {
            throw std::runtime_error("failed to fingerprint asset '" + assetName
                                     + "': " + status.message);
        }
        std::error_code relativeError;
        fs::path logicalName = fs::relative(
            assetPath, fs::absolute(scenePath).parent_path(), relativeError);
        if (relativeError || logicalName.empty())
        {
            logicalName = assetPath.filename();
        }
        assets.addString(logicalName.generic_string()).addU64(content);
    }
    result.assets = assets.value();

    checkpoint::FingerprintBuilder materials;
    materials.addU64(static_cast<std::uint64_t>(scene.materials.size()));
    for (const Material& material : scene.materials)
    {
        materials.addVec3(material.color)
            .addFloat(material.specular.exponent)
            .addVec3(material.specular.color)
            .addFloat(material.hasReflective)
            .addFloat(material.hasRefractive)
            .addFloat(material.indexOfRefraction)
            .addFloat(material.emittance)
            .addI32(material.textureId)
            .addI32(material.bumpTextureId)
            .addI32(material.procedural)
            .addFloat(material.bumpStrength)
            .addFloat(material.checkerScale);
    }
    result.materials = materials.value();

    checkpoint::FingerprintBuilder geometry;
    geometry.addU64(static_cast<std::uint64_t>(scene.geoms.size()));
    for (const Geom& geom : scene.geoms)
    {
        geometry.addI32(static_cast<std::int32_t>(geom.type))
            .addI32(geom.materialid)
            .addVec3(geom.translation)
            .addVec3(geom.rotation)
            .addVec3(geom.scale)
            .addVec3(geom.motionTranslation);
        addMatrix(geometry, geom.transform);
        addMatrix(geometry, geom.inverseTransform);
        addMatrix(geometry, geom.invTranspose);
        for (int vertex = 0; vertex < 3; ++vertex)
        {
            geometry.addVec3(geom.vertices[vertex])
                .addVec3(geom.normals[vertex])
                .addVec2(geom.uvs[vertex]);
        }
        geometry.addBool(geom.hasNormals).addBool(geom.hasUVs);
    }
    geometry.addU64(static_cast<std::uint64_t>(scene.textures.size()));
    for (const TextureDescriptor& texture : scene.textures)
    {
        geometry.addI32(texture.width).addI32(texture.height).addI32(texture.offset);
    }
    geometry.addU64(static_cast<std::uint64_t>(scene.texels.size()));
    for (const glm::vec3& texel : scene.texels)
    {
        geometry.addVec3(texel);
    }
    geometry.addU64(static_cast<std::uint64_t>(scene.bvhNodes.size()));
    for (const BvhNode& node : scene.bvhNodes)
    {
        geometry.addVec3(node.boundsMin).addVec3(node.boundsMax)
            .addI32(node.left).addI32(node.right)
            .addI32(node.first).addI32(node.count);
    }
    geometry.addU64(static_cast<std::uint64_t>(scene.bvhPrimitiveIndices.size()));
    for (int primitive : scene.bvhPrimitiveIndices)
    {
        geometry.addI32(primitive);
    }
    // Shutter endpoints are camera state, but are also part of the canonical
    // parsed scene identity because they change every moving-geometry sample.
    geometry.addFloat(scene.state.camera.shutterOpen)
        .addFloat(scene.state.camera.shutterClose);
    result.geometry = geometry.value();

    checkpoint::FingerprintBuilder renderConfig;
    renderConfig.addU32(config.width)
        .addU32(config.height)
        .addU32(config.maxDepth)
        .addBool(config.antialias)
        .addBool(config.nextEventEstimation)
        .addBool(config.useBvh)
        .addBool(config.compact)
        .addBool(config.materialSort)
        .addBool(config.wavefront)
        .addBool(config.russianRoulette)
        .addU32(config.rrStart)
        .addU32(samplerId)
        .addBool(config.motion);
    result.renderConfig = renderConfig.value();
    return result;
}

std::vector<checkpoint::BvhNodeRecord> checkpointBvh(const Scene& scene)
{
    std::vector<checkpoint::BvhNodeRecord> result;
    result.reserve(scene.bvhNodes.size());
    for (const BvhNode& node : scene.bvhNodes)
    {
        checkpoint::BvhNodeRecord record;
        record.boundsMin = node.boundsMin;
        record.boundsMax = node.boundsMax;
        record.left = node.left;
        record.right = node.right;
        record.first = node.first;
        record.count = node.count;
        result.push_back(record);
    }
    return result;
}

bool sameFloat(float left, float right)
{
    std::uint32_t leftBits = 0;
    std::uint32_t rightBits = 0;
    std::memcpy(&leftBits, &left, sizeof(leftBits));
    std::memcpy(&rightBits, &right, sizeof(rightBits));
    return leftBits == rightBits;
}

bool sameVec3(const glm::vec3& left, const glm::vec3& right)
{
    return sameFloat(left.x, right.x)
        && sameFloat(left.y, right.y)
        && sameFloat(left.z, right.z);
}

void validateCheckpointBvh(const checkpoint::CheckpointData& saved, const Scene& scene)
{
    if (saved.bvhNodes.size() != scene.bvhNodes.size()
        || saved.bvhPrimitiveIndices.size() != scene.bvhPrimitiveIndices.size())
    {
        throw std::runtime_error("checkpoint BVH dimensions do not match rebuilt scene BVH");
    }
    for (std::size_t i = 0; i < saved.bvhNodes.size(); ++i)
    {
        const checkpoint::BvhNodeRecord& left = saved.bvhNodes[i];
        const BvhNode& right = scene.bvhNodes[i];
        if (!sameVec3(left.boundsMin, right.boundsMin)
            || !sameVec3(left.boundsMax, right.boundsMax)
            || left.left != right.left || left.right != right.right
            || left.first != right.first || left.count != right.count)
        {
            throw std::runtime_error("checkpoint BVH differs from rebuilt scene BVH");
        }
    }
    for (std::size_t i = 0; i < saved.bvhPrimitiveIndices.size(); ++i)
    {
        if (saved.bvhPrimitiveIndices[i] != scene.bvhPrimitiveIndices[i])
        {
            throw std::runtime_error("checkpoint BVH primitive order differs from rebuilt scene");
        }
    }
}

checkpoint::ExpectedState expectedState(const ResolvedRun& run,
                                        const Scene& scene,
                                        const fs::path& scenePath)
{
    checkpoint::ExpectedState expected;
    expected.seed = run.renderer.seed;
    expected.samplerVersion = static_cast<std::uint32_t>(run.renderer.sampler);
    expected.config = checkpointConfig(run);
    expected.camera = checkpointCamera(scene.state.camera);
    expected.compatibility = checkpointCompatibility();
    expected.fingerprints = fingerprintsFor(scenePath, scene, expected.config,
        expected.samplerVersion);
    expected.requireSeedAndSampler = true;
    expected.requireCompatibility = true;
    return expected;
}

checkpoint::CheckpointData makeCheckpoint(const Scene& scene,
                                          const checkpoint::ExpectedState& expected,
                                          const std::vector<glm::vec3>& accumulation,
                                          std::uint64_t completedSamples)
{
    checkpoint::CheckpointData data;
    data.completedSamples = completedSamples;
    data.seed = expected.seed;
    data.samplerVersion = expected.samplerVersion;
    data.config = expected.config;
    data.camera = expected.camera;
    data.compatibility = expected.compatibility;
    data.fingerprints = expected.fingerprints;
    data.accumulation = accumulation;
    data.bvhNodes = checkpointBvh(scene);
    data.bvhPrimitiveIndices.reserve(scene.bvhPrimitiveIndices.size());
    for (int primitive : scene.bvhPrimitiveIndices)
    {
        data.bvhPrimitiveIndices.push_back(static_cast<std::int32_t>(primitive));
    }
    return data;
}

void saveCheckpoint(const fs::path& path,
                    const Scene& scene,
                    const checkpoint::ExpectedState& expected,
                    const std::vector<glm::vec3>& accumulation,
                    std::uint64_t completedSamples)
{
    const checkpoint::CheckpointData data = makeCheckpoint(
        scene, expected, accumulation, completedSamples);
    const checkpoint::Status status = checkpoint::saveAtomic(path, data);
    if (!status)
    {
        throw std::runtime_error("checkpoint save failed: " + status.message);
    }
}

ResolvedRun resolveRun(const Options& options,
                       const Scene& scene,
                       const checkpoint::CheckpointMetadata* resumed)
{
    ResolvedRun run;
    const RendererConfig defaults;
    run.width = options.resolution
        ? options.resolution->first
        : resumed ? static_cast<int>(resumed->config.width) : scene.state.camera.resolution.x;
    run.height = options.resolution
        ? options.resolution->second
        : resumed ? static_cast<int>(resumed->config.height) : scene.state.camera.resolution.y;
    const checkpoint::Limits checkpointLimits;
    if (run.width <= 0 || run.height <= 0
        || run.width > std::numeric_limits<int>::max() / 3
        || static_cast<std::uint64_t>(run.width) > checkpointLimits.maxWidth
        || static_cast<std::uint64_t>(run.height) > checkpointLimits.maxHeight
        || static_cast<std::uint64_t>(run.width)
               * static_cast<std::uint64_t>(run.height)
           > std::min(static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                      checkpointLimits.maxPixels))
    {
        throw std::runtime_error("resolved scene resolution is invalid or too large");
    }

    run.renderer.seed = options.seed.value_or(resumed ? resumed->seed : defaults.seed);
    run.renderer.maxDepth = options.depth.value_or(
        resumed ? static_cast<int>(resumed->config.maxDepth)
                : (scene.state.traceDepth > 0 ? scene.state.traceDepth : defaults.maxDepth));
    run.renderer.aa = options.aa.value_or(
        resumed ? resumed->config.antialias : defaults.aa);
    run.renderer.nee = options.nee.value_or(
        resumed ? resumed->config.nextEventEstimation : defaults.nee);
    run.renderer.useBvh = options.bvh.value_or(
        resumed ? resumed->config.useBvh : defaults.useBvh);
    run.renderer.compact = options.compact.value_or(
        resumed ? resumed->config.compact : defaults.compact);
    run.renderer.sort = options.sort.value_or(
        resumed ? resumed->config.materialSort : defaults.sort);
    run.renderer.wavefront = options.wavefront.value_or(
        resumed ? resumed->config.wavefront : defaults.wavefront);
    run.renderer.rr = options.rr.value_or(
        resumed ? resumed->config.russianRoulette : defaults.rr);
    run.renderer.rrStart = options.rrStart.value_or(
        resumed ? static_cast<int>(resumed->config.rrStart) : defaults.rrStart);
    if (options.sampler)
    {
        run.renderer.sampler = *options.sampler;
    }
    else if (resumed)
    {
        if (resumed->samplerVersion ==
                static_cast<std::uint32_t>(RendererSampler::Counter))
        {
            run.renderer.sampler = RendererSampler::Counter;
        }
        else if (resumed->samplerVersion ==
                 static_cast<std::uint32_t>(RendererSampler::Halton))
        {
            run.renderer.sampler = RendererSampler::Halton;
        }
        else
        {
            throw std::runtime_error(
                "checkpoint contains an unknown sampler ID");
        }
    }
    else
    {
        run.renderer.sampler = defaults.sampler;
    }
    run.renderer.motion = options.motion.value_or(
        resumed ? resumed->config.motion : defaults.motion);
    run.renderer.benchmark = options.benchmark;
    run.renderer.debug = options.debug.value_or(RendererDebugMode::None);
    if (run.renderer.maxDepth <= 0 || run.renderer.maxDepth > 4096)
    {
        throw std::runtime_error("resolved trace depth is outside [1, 4096]");
    }
    if (run.renderer.rrStart < 1 || run.renderer.rrStart > 4096)
    {
        throw std::runtime_error("resolved Russian roulette start is outside [1, 4096]");
    }
    if (run.renderer.wavefront && run.renderer.sort)
    {
        throw std::invalid_argument("--sort on requires --wavefront off");
    }

    run.targetSamples = options.samples.value_or(scene.state.iterations);
    run.outputPrefix = options.outputPrefix.value_or(fs::path(scene.state.imageName));
    return run;
}

template<typename Writer>
void writeAtomic(const fs::path& destination, std::ios::openmode mode, Writer&& writer)
{
    fs::path temporary = destination;
    temporary += ".tmp";
    std::error_code error;
    if (fs::exists(temporary, error) && !error)
    {
        throw std::runtime_error("refusing to overwrite stale temporary file: "
                                 + temporary.string());
    }

    try
    {
        std::ofstream output(temporary, std::ios::out | std::ios::trunc | mode);
        if (!output)
        {
            throw std::runtime_error("cannot open output file: " + temporary.string());
        }
        writer(output);
        output.flush();
        if (!output)
        {
            throw std::runtime_error("failed writing output file: " + temporary.string());
        }
        output.close();
        if (!output)
        {
            throw std::runtime_error("failed closing output file: " + temporary.string());
        }
        fs::rename(temporary, destination, error);
        if (error)
        {
            throw std::runtime_error("failed to publish output file '" + destination.string()
                                     + "': " + error.message());
        }
    }
    catch (...)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

class HashedBinaryWriter
{
public:
    explicit HashedBinaryWriter(std::ostream& output) : output_(output) {}

    void bytes(const void* data, std::size_t count)
    {
        output_.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
        if (!output_)
        {
            throw std::runtime_error("binary output write failed");
        }
        checksum_ = checkpoint::fnv1aUpdate(checksum_, data, count);
    }

    void u32(std::uint32_t value)
    {
        const std::array<unsigned char, 4> encoded = {
            static_cast<unsigned char>(value),
            static_cast<unsigned char>(value >> 8u),
            static_cast<unsigned char>(value >> 16u),
            static_cast<unsigned char>(value >> 24u)};
        bytes(encoded.data(), encoded.size());
    }

    void u64(std::uint64_t value)
    {
        std::array<unsigned char, 8> encoded{};
        for (unsigned int byte = 0; byte < encoded.size(); ++byte)
        {
            encoded[byte] = static_cast<unsigned char>(value >> (8u * byte));
        }
        bytes(encoded.data(), encoded.size());
    }

    void floating(float value)
    {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "binary32 float is required");
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }

    std::uint64_t checksum() const { return checksum_; }

    void checksumTrailer()
    {
        const std::uint64_t value = checksum_;
        std::array<unsigned char, 8> encoded{};
        for (unsigned int byte = 0; byte < encoded.size(); ++byte)
        {
            encoded[byte] = static_cast<unsigned char>(value >> (8u * byte));
        }
        output_.write(reinterpret_cast<const char*>(encoded.data()),
                      static_cast<std::streamsize>(encoded.size()));
    }

private:
    std::ostream& output_;
    std::uint64_t checksum_ = checkpoint::kFnv1aOffsetBasis;
};

void validateAccumulation(const std::vector<glm::vec3>& accumulation,
                          std::size_t expectedPixels)
{
    if (accumulation.size() != expectedPixels)
    {
        throw std::runtime_error("renderer returned an unexpected accumulation size");
    }
    for (const glm::vec3& pixel : accumulation)
    {
        if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || !std::isfinite(pixel.z)
            || pixel.x < 0.0f || pixel.y < 0.0f || pixel.z < 0.0f)
        {
            throw std::runtime_error("renderer accumulation contains a non-finite or negative pixel");
        }
    }
}

std::uint64_t writeLinearSums(const fs::path& path,
                              const ResolvedRun& run,
                              std::uint64_t completedSamples,
                              const std::vector<glm::vec3>& accumulation)
{
    std::uint64_t checksum = 0;
    writeAtomic(path, std::ios::binary, [&](std::ostream& output)
    {
        HashedBinaryWriter binary(output);
        constexpr std::array<char, 8> magic = {'C', '5', '6', '5', 'S', 'U', 'M', '\0'};
        binary.bytes(magic.data(), magic.size());
        binary.u32(kSumFileVersion);
        binary.u32(static_cast<std::uint32_t>(run.width));
        binary.u32(static_cast<std::uint32_t>(run.height));
        binary.u32(3); // RGB channels.
        binary.u64(completedSamples);
        for (const glm::vec3& pixel : accumulation)
        {
            binary.floating(pixel.x);
            binary.floating(pixel.y);
            binary.floating(pixel.z);
        }
        checksum = binary.checksum();
        binary.checksumTrailer();
    });
    return checksum;
}

void writeSemanticPng(const fs::path& path, const ResolvedRun& run,
                      std::uint64_t completedSamples,
                      const std::vector<glm::vec3>& accumulation)
{
    const float inverseSamples = 1.0f / static_cast<float>(completedSamples);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(run.width)
        * static_cast<std::size_t>(run.height) * 3U);
    for (int y = 0; y < run.height; ++y)
    {
        for (int x = 0; x < run.width; ++x)
        {
            const std::size_t source = static_cast<std::size_t>(y) * run.width + x;
            const std::size_t destination = (static_cast<std::size_t>(y) * run.width
                + (run.width - 1 - x)) * 3U;
            const glm::vec3 value = glm::clamp(
                accumulation[source] * inverseSamples, 0.0f, 1.0f);
            for (int channel = 0; channel < 3; ++channel)
            {
                bytes[destination + static_cast<std::size_t>(channel)] =
                    static_cast<unsigned char>(std::lround(value[channel] * 255.0f));
            }
        }
    }

    fs::path temporary = path;
    temporary += ".tmp";
    std::error_code error;
    if (fs::exists(temporary, error) && !error)
    {
        throw std::runtime_error("refusing to overwrite stale PNG temporary file: "
            + temporary.string());
    }
    if (error)
    {
        throw std::runtime_error("cannot inspect PNG temporary file: "
            + temporary.string());
    }
    if (stbi_write_png(temporary.string().c_str(), run.width, run.height, 3,
            bytes.data(), run.width * 3) == 0)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw std::runtime_error("failed to write semantic PNG image: "
            + path.string());
    }
    fs::rename(temporary, path, error);
    if (error)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw std::runtime_error("failed to publish semantic PNG image '"
            + path.string() + "': " + error.message());
    }
}

void writePng(const fs::path& prefix,
              const ResolvedRun& run,
              std::uint64_t completedSamples,
              const std::vector<glm::vec3>& accumulation)
{
    if (completedSamples == 0)
    {
        throw std::runtime_error("cannot make a display image with zero samples");
    }
    if (run.renderer.debug != RendererDebugMode::None)
    {
        writeSemanticPng(suffixed(prefix, ".png"), run, completedSamples,
            accumulation);
        return;
    }
    Image image(run.width, run.height);
    const float inverseSamples = 1.0f / static_cast<float>(completedSamples);
    for (int y = 0; y < run.height; ++y)
    {
        for (int x = 0; x < run.width; ++x)
        {
            const std::size_t source = static_cast<std::size_t>(y) * run.width + x;
            // The starter preview's camera convention requires this horizontal
            // display flip. The lossless sum file remains in renderer order.
            image.setPixel(run.width - 1 - x, y, accumulation[source] * inverseSamples);
        }
    }
    image.savePNG(prefix.string());
}

void writeBenchmark(const fs::path& path,
                    const std::vector<RendererBenchmarkRow>& rows,
                    const RendererBenchmarkSummary& summary,
                    double renderLoopWallMs)
{
    writeAtomic(path, std::ios::openmode(0), [&](std::ostream& output)
    {
        output << "record_type,sample,bounce,active_before,active_after,"
                  "intersection_ms,sort_ms,shade_ms,compaction_ms,"
                  "count_ms,host_ms,gpu_ms,measured_samples,render_loop_wall_ms,"
                  "queue_ms,queue_host_ms,diffuse_ms,mirror_ms,dielectric_ms,"
                  "emission_ms,terminal_ms,queued_diffuse,queued_mirror,"
                  "queued_dielectric,queued_emission,queued_terminal,"
                  "kernel_dispatches,rr_tested,rr_terminated\n";
        output << std::setprecision(9);
        for (const RendererBenchmarkRow& row : rows)
        {
            output << "bounce," << row.sample << ',' << row.bounce << ','
                   << row.activeBefore << ',' << row.activeAfter << ','
                   << row.intersectionMs << ',' << row.sortMs << ','
                   << row.shadeMs << ',' << row.compactionMs << ','
                   << row.countMs << ",,,,," << row.queueMs << ','
                   << row.queueHostMs << ',' << row.diffuseMs << ','
                   << row.mirrorMs << ',' << row.dielectricMs << ','
                   << row.emissionMs << ',' << row.terminalMs << ','
                   << row.queuedDiffuse << ',' << row.queuedMirror << ','
                   << row.queuedDielectric << ',' << row.queuedEmission << ','
                   << row.queuedTerminal << ',' << row.kernelDispatches << ','
                   << row.rrTested << ',' << row.rrTerminated << '\n';
        }
        output << "overall,,,,,,,,,," << summary.hostMilliseconds << ','
               << summary.gpuMilliseconds << ',' << summary.measuredSamples << ','
               << renderLoopWallMs << ",,,,,,,,,,,,,,,\n";
    });
}

json configJson(const ResolvedRun& run)
{
    return {
        {"max_depth", run.renderer.maxDepth},
        {"aa", run.renderer.aa},
        {"nee", run.renderer.nee},
        {"bvh", run.renderer.useBvh},
        {"compact", run.renderer.compact},
        {"sort", run.renderer.sort},
        {"wavefront", run.renderer.wavefront},
        {"rr", run.renderer.rr},
        {"rr_start", run.renderer.rrStart},
        {"sampler", samplerName(run.renderer.sampler)},
        {"sampler_id", static_cast<std::uint32_t>(run.renderer.sampler)},
        {"motion", run.renderer.motion},
        {"debug", debugModeName(run.renderer.debug)},
        {"depth_semantics", "maximum surface intersections; final intersection is emission-only"}};
}

json compatibilityJson(const checkpoint::CompatibilityInfo& compatibility)
{
    return {
        {"cuda_runtime_version", compatibility.cudaRuntimeVersion},
        {"cuda_driver_version", compatibility.cudaDriverVersion},
        {"compute_capability", std::to_string(compatibility.computeCapabilityMajor)
                                   + "." + std::to_string(compatibility.computeCapabilityMinor)},
        {"device", compatibility.deviceIdentifier},
        {"baseline_starter_revision", compatibility.sourceRevision},
        {"build_identifier", compatibility.buildIdentifier}};
}

void writeMetadata(const fs::path& path,
                   const Options& options,
                   const ResolvedRun& run,
                   const Scene& scene,
                   std::uint64_t completedSamples,
                   std::uint64_t sumChecksum,
                   const checkpoint::ExpectedState& expected,
                   const std::optional<RendererBenchmarkSummary>& benchmark,
                   double renderLoopWallMs,
                   double lastSampleWallMs)
{
    json metadata;
    metadata["format_version"] = 1;
    metadata["scene"] = options.scenePath.string();
    metadata["resolution"] = {run.width, run.height};
    metadata["target_samples"] = run.targetSamples;
    metadata["completed_samples"] = completedSamples;
    metadata["seed"] = run.renderer.seed;
    metadata["sampler_version"] = expected.samplerVersion;
    metadata["sampler"] = samplerName(run.renderer.sampler);
    metadata["sampler_id"] = expected.samplerVersion;
    metadata["config"] = configJson(run);
    if (run.renderer.debug == RendererDebugMode::None)
    {
        metadata["debug_view"] = nullptr;
    }
    else
    {
        metadata["debug_view"] = {
            {"mode", debugModeName(run.renderer.debug)},
            {"intersection", "first camera-ray surface; misses are black"},
            {"normals_encoding", "0.5 * post-bump oriented shading normal + 0.5"},
            {"uv_encoding", "RGB = (fract(u), fract(v), 0)"},
            {"depth_encoding", "RGB = t / (1 + t)"},
            {"png_encoding", "averaged semantic values clamped to [0,1], no tone or transfer curve"}};
    }
    metadata["camera"] = {
        {"position", {scene.state.camera.position.x, scene.state.camera.position.y,
                      scene.state.camera.position.z}},
        {"look_at", {scene.state.camera.lookAt.x, scene.state.camera.lookAt.y,
                     scene.state.camera.lookAt.z}},
        {"aperture", scene.state.camera.aperture},
        {"focus_distance", scene.state.camera.focusDistance},
        {"shutter", {scene.state.camera.shutterOpen,
                     scene.state.camera.shutterClose}}};
    metadata["outputs"] = {
        {"png", suffixed(run.outputPrefix, ".png").string()},
        {"linear_sum", suffixed(run.outputPrefix, ".sum.bin").string()}};
    metadata["display_transform"] = {
        {"normalization", "divide accumulated RGB sums by completed_samples"},
        {"tone_mapping", run.renderer.debug == RendererDebugMode::None
            ? "Reinhard c/(1+c)" : "none (semantic values clamped to [0,1])"},
        {"transfer", run.renderer.debug == RendererDebugMode::None
            ? "IEC 61966-2-1 sRGB" : "none (linear semantic encoding)"},
        {"orientation", "horizontal flip matching the starter preview camera convention"}};
    metadata["linear_sum_format"] = {
        {"magic", "C565SUM\\0"},
        {"version", kSumFileVersion},
        {"endianness", "little"},
        {"header_bytes", 32},
        {"payload_offset_bytes", 32},
        {"checksum_trailer_bytes", 8},
        {"header_fields", "magic[8], u32 version,width,height,channels, u64 completed_samples"},
        {"payload", "row-major renderer-order interleaved RGB IEEE-754 binary32 accumulated sums"},
        {"png_orientation", "horizontal flip of renderer-order pixels"},
        {"trailer", "little-endian FNV-1a-64 over header and payload"},
        {"checksum", sumChecksum},
        {"checksum_security", "change/corruption detection only; not cryptographic"}};
    metadata["fingerprints"] = {
        {"scene_file", expected.fingerprints.sceneFile},
        {"assets", expected.fingerprints.assets},
        {"materials", expected.fingerprints.materials},
        {"geometry", expected.fingerprints.geometry},
        {"render_config", expected.fingerprints.renderConfig}};
    metadata["compatibility"] = compatibilityJson(expected.compatibility);
    metadata["resumed_from"] = options.resumePath
        ? json(options.resumePath->string()) : json(nullptr);
    metadata["checkpoint"] = options.checkpointPath
        ? json(options.checkpointPath->string()) : json(nullptr);
    metadata["checkpoint_every"] = options.checkpointEvery
        ? json(*options.checkpointEvery) : json(nullptr);
    metadata["benchmark_enabled"] = options.benchmark;
    metadata["benchmark_instrumentation"] = options.benchmark
        ? json("shade_ms is aggregate shader time in both scheduling modes. diffuse_ms, mirror_ms, dielectric_ms, emission_ms, and terminal_ms are an alternative wavefront breakdown and must not be added to shade_ms. queue_host_ms is host queue bookkeeping/submission time, not GPU stage time. count_ms is the compact=off active-path count.")
        : json(nullptr);
    metadata["render_loop_wall_ms"] = renderLoopWallMs;
    metadata["run_policy"] = {
        {"time_budget_ms", options.timeBudgetMs
            ? json(*options.timeBudgetMs) : json(nullptr)},
        {"stop_reason", options.timeBudgetMs
                && completedSamples < run.targetSamples
            ? "time_budget" : "sample_target"},
        {"last_sample_wall_ms", lastSampleWallMs}};
    if (benchmark)
    {
        metadata["benchmark"] = {
            {"csv", suffixed(run.outputPrefix, ".benchmark.csv").string()},
            {"measured_samples", benchmark->measuredSamples},
            {"renderer_host_ms", benchmark->hostMilliseconds},
            {"renderer_gpu_ms", benchmark->gpuMilliseconds}};
    }

    writeAtomic(path, std::ios::openmode(0), [&](std::ostream& output)
    {
        output << std::setw(2) << metadata << '\n';
    });
}

int runHeadless(const Options& options)
{
    requireRegularInput(options.scenePath, "scene");
    if (options.resumePath)
    {
        requireRegularInput(*options.resumePath, "resume checkpoint");
    }

    checkpoint::CheckpointMetadata resumeMetadata;
    checkpoint::CheckpointMetadata* resumeMetadataPointer = nullptr;
    if (options.resumePath)
    {
        const checkpoint::Status status = checkpoint::inspect(
            *options.resumePath, resumeMetadata);
        if (!status)
        {
            throw std::runtime_error("checkpoint preflight failed: " + status.message);
        }
        resumeMetadataPointer = &resumeMetadata;
    }

    std::optional<CoutRedirect> validationLogging;
    if (options.validate)
    {
        validationLogging.emplace(std::cerr);
    }

    Scene scene(options.scenePath.string());
    ResolvedRun run = resolveRun(options, scene, resumeMetadataPointer);
    configureCameraResolution(scene, run.width, run.height);
    scene.state.traceDepth = run.renderer.maxDepth;

    if (!options.validate)
    {
        validateUniqueOutputs(
            run.outputPrefix, options.benchmark, options.checkpointPath);
        if (options.checkpointPath)
        {
            probeWritableSibling(*options.checkpointPath);
        }
    }

    const checkpoint::ExpectedState expected = expectedState(run, scene, options.scenePath);
    checkpoint::CheckpointData resumed;
    if (options.resumePath)
    {
        const checkpoint::Status status = checkpoint::load(
            *options.resumePath, resumed, &expected);
        if (!status)
        {
            throw std::runtime_error("checkpoint continuation rejected: " + status.message);
        }
        validateCheckpointBvh(resumed, scene);
        if (run.targetSamples < resumed.completedSamples)
        {
            throw std::runtime_error("--samples is below the checkpoint's completed sample count");
        }
    }

    pathtraceSetConfig(run.renderer);
    RendererLifetime renderer;
    renderer.initialize(scene);
    if (options.resumePath)
    {
        pathtraceUploadAccumulation(resumed.accumulation, resumed.completedSamples);
    }

    if (options.validate)
    {
        const RendererValidationReport report = pathtraceValidate();
        renderer.shutdown();
        validationLogging.reset();
        std::cout << pathtraceValidationJson(report) << '\n';
        return report.passed ? 0 : 2;
    }

    std::uint64_t completed = pathtraceSampleCount();
    if (completed != (options.resumePath ? resumed.completedSamples : 0u))
    {
        throw std::runtime_error("renderer sample counter was not initialized as requested");
    }

    std::vector<glm::vec3> accumulation;
    std::uint64_t cachedAtSample = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t checkpointedAtSample = std::numeric_limits<std::uint64_t>::max();
    double lastSampleWallMs = 0.0;
    const auto renderStarted = std::chrono::steady_clock::now();
    while (completed < run.targetSamples)
    {
        const auto sampleStarted = std::chrono::steady_clock::now();
        pathtraceRunSample();
        const auto sampleStopped = std::chrono::steady_clock::now();
        lastSampleWallMs = std::chrono::duration<double, std::milli>(
            sampleStopped - sampleStarted).count();
        const std::uint64_t next = pathtraceSampleCount();
        if (next != completed + 1)
        {
            throw std::runtime_error("renderer did not advance by exactly one completed sample");
        }
        completed = next;
        cachedAtSample = std::numeric_limits<std::uint64_t>::max();

        if (options.checkpointPath && options.checkpointEvery
            && completed % *options.checkpointEvery == 0)
        {
            pathtraceDownloadAccumulation(accumulation);
            cachedAtSample = completed;
            saveCheckpoint(*options.checkpointPath, scene, expected,
                           accumulation, completed);
            checkpointedAtSample = completed;
        }

        // The budget is a diagnostic stopping policy, checked only between
        // complete samples. It is deliberately absent from renderer identity,
        // scene fingerprints, and checkpoint state.
        if (options.timeBudgetMs && completed < run.targetSamples)
        {
            const double elapsedMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - renderStarted).count();
            if (elapsedMs >= static_cast<double>(*options.timeBudgetMs))
            {
                break;
            }
        }
    }
    const auto renderStopped = std::chrono::steady_clock::now();
    const double renderLoopWallMs = std::chrono::duration<double, std::milli>(
        renderStopped - renderStarted).count();

    if (cachedAtSample != completed)
    {
        pathtraceDownloadAccumulation(accumulation);
        cachedAtSample = completed;
    }
    const std::size_t pixelCount = static_cast<std::size_t>(run.width)
        * static_cast<std::size_t>(run.height);
    validateAccumulation(accumulation, pixelCount);

    if (options.checkpointPath && checkpointedAtSample != completed)
    {
        saveCheckpoint(*options.checkpointPath, scene, expected,
                       accumulation, completed);
    }

    const fs::path sumPath = suffixed(run.outputPrefix, ".sum.bin");
    const std::uint64_t sumChecksum = writeLinearSums(
        sumPath, run, completed, accumulation);
    writePng(run.outputPrefix, run, completed, accumulation);

    std::optional<RendererBenchmarkSummary> benchmarkSummary;
    if (options.benchmark)
    {
        const std::vector<RendererBenchmarkRow> rows = pathtraceTakeBenchmarkRows();
        benchmarkSummary = pathtraceBenchmarkSummary();
        writeBenchmark(suffixed(run.outputPrefix, ".benchmark.csv"), rows,
                       *benchmarkSummary, renderLoopWallMs);
    }

    writeMetadata(suffixed(run.outputPrefix, ".json"), options, run, scene,
                  completed, sumChecksum, expected, benchmarkSummary,
                  renderLoopWallMs, lastSampleWallMs);
    renderer.shutdown();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = parseOptions(argc, argv);
        if (options.help)
        {
            std::cout << usage(argc > 0 ? argv[0] : "cis565_path_tracer_headless");
            return 0;
        }
        return runHeadless(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << usage(argc > 0 ? argv[0] : "cis565_path_tracer_headless");
        return 1;
    }
}
