#include "app/cli.h"

#include "utilities.h"

#include <climits>
#include <cstdio>
#include <cstdlib>

namespace
{
const char* const USAGE =
    "Usage: %s SCENE [--headless] [--spp N] [--res WxH] [--depth N] [--out PATH.png] [--env FILE]\n"
    "                     [--no-rr] [--no-aa] [--sort] [--no-compact] [--no-nee] [--no-env-nee] [--no-env-compensation]\n"
    "                     [--tonemap none|aces|agx|agx-punchy] [--exposure X]\n"
    "                     [--lens pinhole|thin|FILE] [--aperture R] [--focus D] [--fovy DEG] [--units-per-meter U]\n"
    "                     [--no-exit-pupil] [--camera-weight simple|physical]\n"
    "                     [--no-optix] [--optix-validate] [--timing]\n"
    "       %s --list\n"
    "  SCENE            a scene .json, a .gltf / .glb file that is the whole scene, or a scene name:\n"
    "                   scenes/<name>.json or an entry of scenes/catalog.json\n"
    "  --list           print the scene names and exit\n"
    "  --headless       render without a window and exit after saving\n"
    "  --spp N          override the scene's ITERATIONS\n"
    "  --res WxH        override the scene's RES\n"
    "  --depth N        override the scene's DEPTH, the most rays a path may trace\n"
    "  --out PATH       write exactly this file (default: img/auto_saved/<FILE>.<time>.<spp>samp.png);\n"
    "                   a .hdr or .exr path keeps the scene-linear average, without view transform or\n"
    "                   exposure; .exr is exact float32, .hdr truncates to 8-bit mantissas\n"
    "  --env FILE       light the scene with this lat-long .hdr or .exr instead of its own environment\n"
    "  --lens TYPE      the camera: pinhole (the scene's default), thin (a thin lens), or a lens file\n"
    "                   in scenes/lenses/ (a .json prescription) for the real lens\n"
    "  --aperture R     the thin lens's aperture radius, scene units (0 is a pinhole)\n"
    "  --focus D        the thin or real lens's focus distance, scene units (default: the distance to LOOKAT)\n"
    "  --fovy DEG       override the scene's FOVY (the real lens sets its own view from the film size)\n"
    "  --units-per-meter U  override the scene's UNITS_PER_METER, which sizes the real lens against the scene\n"
    "  --no-exit-pupil  real lens: aim camera rays at the whole rear element instead of the exit pupil table\n"
    "  --camera-weight  real lens: simple (default; the center pixel wide open weighs 1) or physical\n"
    "                   (irradiance on the film, which needs an exposure to look right)\n"
    "  --no-rr          disable Russian roulette path termination\n"
    "  --no-aa          no antialiasing: every camera ray goes through its pixel's center\n"
    "  --sort           sort paths by material before shading (off by default: it costs 1.3 to 2x, see the README)\n"
    "  --no-sort        the default, kept for scripts\n"
    "  --no-compact     keep ended paths in place instead of compacting them away, for comparisons\n"
    "  --no-nee         no next event estimation: lights count only when a path hits them\n"
    "  --no-env-nee     leave the environment out of the light list: paths find it by BSDF sampling only\n"
    "  --no-env-compensation  sample the environment map by its radiance alone, without MIS compensation\n"
    "  --no-optix       intersect with the naive per-object kernel instead of OptiX (implies --no-nee)\n"
    "  --optix-validate OptiX validation mode: checks every launch, slow\n"
    "  --timing         with --headless: print load, init and per-bounce stage times as CSV lines\n"
    "  --tonemap MODE   view transform for display and PNG (default agx-punchy; none = raw clamp)\n"
    "  --exposure X     linear multiplier before the view transform (default 1.0)\n";
}  // namespace

Options parseArguments(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto needValue = [&](const char* flag) -> const char* {
            if (i + 1 >= argc)
            {
                fatal("%s needs a value", flag);
            }
            return argv[++i];
        };
        auto needPositiveInt = [&](const char* flag) -> int {
            const char* value = needValue(flag);
            char* end = nullptr;
            const long n = strtol(value, &end, 10);
            if (end == value || *end != '\0' || n <= 0 || n > INT_MAX)
            {
                fatal("%s expects a positive whole number, not \"%s\"", flag, value);
            }
            return (int)n;
        };
        if (a == "--headless")
        {
            options.headless = true;
        }
        else if (a == "--list")
        {
            options.list = true;
        }
        else if (a == "--spp")
        {
            options.overrides.iterations = needPositiveInt("--spp");
        }
        else if (a == "--res")
        {
            const char* value = needValue("--res");
            SceneOverrides& ov = options.overrides;
            if (sscanf(value, "%dx%d", &ov.width, &ov.height) != 2 || ov.width <= 0 || ov.height <= 0)
            {
                fatal("--res expects WxH, e.g. 800x600, not \"%s\"", value);
            }
        }
        else if (a == "--depth")
        {
            options.overrides.traceDepth = needPositiveInt("--depth");
        }
        else if (a == "--out")
        {
            options.outPath = needValue("--out");
        }
        else if (a == "--env")
        {
            options.overrides.environmentFile = needValue("--env");
        }
        else if (a == "--lens")
        {
            options.overrides.lens = needValue("--lens");
        }
        else if (a == "--aperture")
        {
            const char* value = needValue("--aperture");
            char* end = nullptr;
            options.overrides.apertureRadius = strtof(value, &end);
            if (end == value || *end != '\0' || !(options.overrides.apertureRadius >= 0.0f))
            {
                fatal("--aperture expects a number of 0 or more, not \"%s\"", value);
            }
        }
        else if (a == "--focus")
        {
            const char* value = needValue("--focus");
            char* end = nullptr;
            options.overrides.focusDistance = strtof(value, &end);
            if (end == value || *end != '\0' || !(options.overrides.focusDistance > 0.0f))
            {
                fatal("--focus expects a positive number, not \"%s\"", value);
            }
        }
        else if (a == "--fovy")
        {
            const char* value = needValue("--fovy");
            char* end = nullptr;
            options.overrides.fovy = strtof(value, &end);
            if (end == value || *end != '\0' || !(options.overrides.fovy > 0.0f && options.overrides.fovy < 180.0f))
            {
                fatal("--fovy expects degrees between 0 and 180, not \"%s\"", value);
            }
        }
        else if (a == "--units-per-meter")
        {
            const char* value = needValue("--units-per-meter");
            char* end = nullptr;
            options.overrides.unitsPerMeter = strtof(value, &end);
            if (end == value || *end != '\0' || !(options.overrides.unitsPerMeter > 0.0f))
            {
                fatal("--units-per-meter expects a positive number, not \"%s\"", value);
            }
        }
        else if (a == "--no-rr")
        {
            options.russianRoulette = false;
        }
        else if (a == "--no-aa")
        {
            options.antialiasing = false;
        }
        else if (a == "--sort")
        {
            options.materialSort = true;
        }
        else if (a == "--no-sort")
        {
            options.materialSort = false;
        }
        else if (a == "--no-compact")
        {
            options.compaction = false;
        }
        else if (a == "--no-exit-pupil")
        {
            options.exitPupil = false;
        }
        else if (a == "--camera-weight")
        {
            std::string m = needValue("--camera-weight");
            if (m == "simple")        options.physicalCameraWeight = false;
            else if (m == "physical") options.physicalCameraWeight = true;
            else fatal("--camera-weight expects simple or physical, not \"%s\"", m.c_str());
        }
        else if (a == "--no-nee")
        {
            options.nee = false;
        }
        else if (a == "--no-env-nee")
        {
            options.overrides.environmentLight = false;
        }
        else if (a == "--no-env-compensation")
        {
            options.overrides.environmentCompensation = false;
        }
        else if (a == "--no-optix")
        {
            options.optix = false;
        }
        else if (a == "--optix-validate")
        {
            options.optixValidation = true;
        }
        else if (a == "--timing")
        {
            options.timing = true;
        }
        else if (a == "--tonemap")
        {
            std::string m = needValue("--tonemap");
            if (m == "none")      options.toneMap = TONEMAP_NONE;
            else if (m == "aces") options.toneMap = TONEMAP_ACES;
            else if (m == "agx")  options.toneMap = TONEMAP_AGX;
            else if (m == "agx-punchy") options.toneMap = TONEMAP_AGX_PUNCHY;
            else
            {
                fatal("--tonemap expects none, aces, agx or agx-punchy, not \"%s\"", m.c_str());
            }
        }
        else if (a == "--exposure")
        {
            const char* value = needValue("--exposure");
            char* end = nullptr;
            options.exposure = strtof(value, &end);
            if (end == value || *end != '\0' || !(options.exposure > 0.0f))
            {
                fatal("--exposure expects a positive number, not \"%s\"", value);
            }
        }
        else if (a.size() > 2 && a.substr(0, 2) == "--")
        {
            fprintf(stderr, USAGE, argv[0], argv[0]);
            fatal("unknown option %s", argv[i]);
        }
        else
        {
            options.sceneFile = argv[i];
        }
    }

    if (options.sceneFile.empty() && !options.list)
    {
        fprintf(stderr, USAGE, argv[0], argv[0]);
        fatal("no scene given");
    }
    return options;
}
