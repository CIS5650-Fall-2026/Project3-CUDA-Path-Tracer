#include "scene/lens.h"

#include "render/lens_trace.h"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

const std::string LENS_DIR = "scenes/lenses/";

namespace
{
[[noreturn]] void lensError(const std::string& name, const char* format, ...)
{
    char message[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof message, format, args);
    va_end(args);
    throw std::runtime_error(name + ": " + message);
}

std::string lowercaseExtensionOf(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext)
    {
        c = (char)std::tolower((unsigned char)c);
    }
    return ext;
}

void validate(LensSystem& lens, const std::string& name)
{
    if (lens.surfaces.empty())
    {
        lensError(name, "no surfaces");
    }
    lens.stopIndex = -1;
    for (size_t i = 0; i < lens.surfaces.size(); ++i)
    {
        const LensSurface& s = lens.surfaces[i];
        if (s.isStop)
        {
            if (lens.stopIndex >= 0)
            {
                lensError(name, "surfaces %d and %d are both the stop", lens.stopIndex + 1, (int)i + 1);
            }
            lens.stopIndex = (int)i;
        }
        // Only the last gap may be 0: the film distance is not known.
        const bool last = i + 1 == lens.surfaces.size();
        if (s.thickness < 0.0f || (!last && s.thickness == 0.0f))
        {
            lensError(name, "surface %d has thickness %g", (int)i + 1, s.thickness);
        }
        if (!(s.apertureRadius > 0.0f))
        {
            lensError(name, "surface %d has no clear aperture", (int)i + 1);
        }
        if (s.ior < 1.0f)
        {
            lensError(name, "surface %d has index %g, below air", (int)i + 1, s.ior);
        }
    }
    if (lens.stopIndex < 0)
    {
        lensError(name, "no aperture stop (a surface with \"stop\": true)");
    }
}
}  // namespace

std::string findLensFile(const std::string& argument)
{
    if (std::filesystem::is_regular_file(argument))
    {
        return argument;
    }
    const std::string inFolder = LENS_DIR + argument;
    for (const char* ext : {"", ".json"})
    {
        if (std::filesystem::is_regular_file(inFolder + ext))
        {
            return inFolder + ext;
        }
    }
    throw std::runtime_error("no lens file " + argument + " (tried " + argument + ", " + inFolder + " and " + inFolder
        + ".json)");
}

std::vector<std::string> lensFiles()
{
    std::vector<std::string> files;
    if (!std::filesystem::is_directory(LENS_DIR))
    {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(LENS_DIR))
    {
        if (entry.is_regular_file() && lowercaseExtensionOf(entry.path().string()) == ".json")
        {
            files.push_back(entry.path().generic_string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

LensSystem loadLens(const std::string& argument)
{
    const std::string file = findLensFile(argument);
    std::ifstream in(file, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(file + ": cannot open");
    }
    std::stringstream text;
    text << in.rdbuf();
    if (lowercaseExtensionOf(file) != ".json")
    {
        throw std::runtime_error(file + ": not a lens file (.json)");
    }
    LensSystem lens = parseLensJson(text.str(), file);
    lens.file = file;
    analyzeLens(lens);
    return lens;
}

void analyzeLens(LensSystem& lens)
{
    const LensSurface* s = lens.surfaces.data();
    const int count = (int)lens.surfaces.size();
    float zStop = 0.0f;
    float zLast = 0.0f;
    for (int i = 0; i + 1 < count; ++i)
    {
        zLast += s[i].thickness;
        if (i < lens.stopIndex)
        {
            zStop += s[i].thickness;
        }
    }

    // A near-axis parallel ray: its axis crossing is the paraxial focus, height over exit slope the focal length.
    const float h = 0.01f * s[0].apertureRadius;
    glm::vec3 o(h, 0.0f, -1.0f);
    glm::vec3 d(0.0f, 0.0f, 1.0f);
    if (!traceLens(s, 0, count - 1, 0.0f, o, d) || d.x >= 0.0f)
    {
        throw std::runtime_error(lens.name + ": a ray along the axis does not come to a focus behind the lens");
    }
    LensSystem::Paraxial& p = lens.paraxial;
    const float zFocus = o.z - o.x * d.z / d.x;
    p.focalLength = -h * d.z / d.x;
    p.backFocus = zFocus - zLast;
    if (lens.surfaces.back().thickness == 0.0f)
    {
        lens.surfaces.back().thickness = p.backFocus;
    }
    p.filmDistance = lensLength(lens);

    // A slightly leaning ray from the stop's center, traced out the front, crosses the axis at the pupil's center.
    if (lens.stopIndex == 0)
    {
        p.entrancePupilZ = 0.0f;
    }
    else
    {
        o = glm::vec3(0.0f, 0.0f, zStop);
        d = glm::normalize(glm::vec3(0.001f, 0.0f, -1.0f));
        if (!traceLens(s, lens.stopIndex - 1, 0, zStop - s[lens.stopIndex - 1].thickness, o, d) || d.x == 0.0f)
        {
            throw std::runtime_error(lens.name + ": a ray from the stop's center does not get out of the front");
        }
        p.entrancePupilZ = o.z - o.x * d.z / d.x;
    }

    // Pupil radius: bisection for the widest parallel ray that gets through.
    float lo = 0.0f;
    float hi = s[0].apertureRadius;
    for (int i = 0; i < 40; ++i)
    {
        const float mid = 0.5f * (lo + hi);
        o = glm::vec3(mid, 0.0f, -1.0f);
        d = glm::vec3(0.0f, 0.0f, 1.0f);
        (traceLens(s, 0, count - 1, 0.0f, o, d) ? lo : hi) = mid;
    }
    p.entrancePupilRadius = lo;
    p.fNumber = p.focalLength / (2.0f * p.entrancePupilRadius);
}

LensSystem parseLensJson(const std::string& text, const std::string& name)
{
    json data;
    try
    {
        data = json::parse(text);
    }
    catch (const std::exception& e)
    {
        lensError(name, "%s", e.what());
    }
    LensSystem lens;
    try
    {
        lens.name = data.value("name", std::filesystem::path(name).stem().string());
        lens.source = data.value("source", name);
        lens.apertureSource = data.value("apertures", std::string("the lens file"));
        lens.focalLength = data.value("focalLength", 0.0f);
        lens.fNumber = data.value("fNumber", 0.0f);
        lens.blades = data.value("blades", 0);
        lens.closeFocusMagnification = data.value("closeFocusMagnification", 0.0f);
        if (data.contains("sensor"))
        {
            const json& sensor = data.at("sensor");
            if (!sensor.is_array() || sensor.size() != 2)
            {
                lensError(name, "sensor is not [width, height]");
            }
            lens.sensorMm = glm::vec2(sensor.at(0).get<float>(), sensor.at(1).get<float>());
        }
        const json& surfaces = data.at("surfaces");
        if (!surfaces.is_array())
        {
            lensError(name, "surfaces is not an array");
        }
        for (size_t i = 0; i < surfaces.size(); ++i)
        {
            const json& j = surfaces.at(i);
            LensSurface s = {};
            s.radius = j.at("radius").get<float>();
            s.thickness = j.at("thickness").get<float>();
            s.ior = j.value("ior", 1.0f);
            s.abbe = j.value("abbe", 0.0f);
            s.apertureRadius = 0.5f * j.at("aperture").get<float>();
            s.isStop = j.value("stop", false) ? 1 : 0;
            s.conic = j.value("conic", 0.0f);
            if (j.contains("aspheric"))
            {
                const json& a = j.at("aspheric");
                if (!a.is_array())
                {
                    lensError(name, "surface %d: aspheric is not an array [A4, A6, ...]", (int)i + 1);
                }
                s.asphericOffset = (int)lens.aspheric.size();
                s.asphericCount = (int)a.size();
                for (size_t k = 0; k < a.size(); ++k)
                {
                    lens.aspheric.push_back(a.at(k).get<float>());
                }
            }
            if (j.contains("focus"))
            {
                const json& f = j.at("focus");
                if (!f.is_array() || f.size() != 2)
                {
                    lensError(name, "surface %d: focus is not [gap at infinity, gap at the closest focus]", (int)i + 1);
                }
                if (lens.focusSurface >= 0)
                {
                    lensError(name, "surfaces %d and %d both have a focus gap; one is supported", lens.focusSurface + 1, (int)i + 1);
                }
                lens.focusSurface = (int)i;
                lens.focusGap = glm::vec2(f.at(0).get<float>(), f.at(1).get<float>());
                s.thickness = lens.focusGap.x;  // the table is at infinity
            }
            lens.surfaces.push_back(s);
        }
    }
    catch (const json::exception& e)
    {
        lensError(name, "%s", e.what());
    }
    validate(lens, name);
    return lens;
}

float lensLength(const LensSystem& lens)
{
    float z = 0.0f;
    for (const LensSurface& s : lens.surfaces)
    {
        z += s.thickness;
    }
    return z;
}

float lensFilmDistance(const LensSystem& lens, float objectZ)
{
    const LensSurface* s = lens.surfaces.data();
    const int count = (int)lens.surfaces.size();
    if (objectZ >= 0.0f || objectZ < -1e7f)  // beyond 10 km the object is at infinity
    {
        return lens.paraxial.filmDistance;
    }
    // The near-axis ray from the object point, started 1 mm in front of the lens to keep the trace in float range
    const float h = 0.01f * s[0].apertureRadius;
    const float slope = h / -objectZ;
    glm::vec3 o(h - slope, 0.0f, -1.0f);
    glm::vec3 d = glm::normalize(glm::vec3(slope, 0.0f, 1.0f));
    if (!traceLens(s, 0, count - 1, 0.0f, o, d) || d.x >= 0.0f)
    {
        return lens.paraxial.filmDistance;
    }
    return o.z - o.x * d.z / d.x;
}

float lensAimRadius(const LensSystem& lens, float halfDiagonal)
{
    const LensSurface& rear = lens.surfaces.back();
    const float a = rear.apertureRadius;
    const float R = std::fabs(rear.radius);
    const float sag = R == 0.0f ? 0.0f : R - std::sqrt(std::max(R * R - a * a, 0.0f));
    return a + sag * (halfDiagonal + a) / rear.thickness;
}
