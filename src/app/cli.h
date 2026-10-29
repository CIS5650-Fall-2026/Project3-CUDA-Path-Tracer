#pragma once

#include "scene/scene.h"
#include "render/tonemap.h"

#include <string>

// The parsed command line.
struct Options
{
    std::string sceneFile;       // a path or a scene name
    bool list = false;
    SceneOverrides overrides;
    bool headless = false;
    std::string outPath;         // empty: img/auto_saved/<FILE>.<time>.<spp>samp.png
    ToneMapMode toneMap = TONEMAP_AGX_PUNCHY;
    float exposure = 1.0f;
    bool russianRoulette = true;
    bool antialiasing = true;
    // Off by default: the sort's gather costs more than it saves the shade kernel (README, Performance).
    bool materialSort = false;
    bool compaction = true;
    bool exitPupil = true;
    bool physicalCameraWeight = false;
    bool nee = true;
    bool optix = true;
    bool optixValidation = false;
    bool timing = false;
};

// Parses the arguments; a bad option or value ends the program with the reason.
Options parseArguments(int argc, char** argv);
