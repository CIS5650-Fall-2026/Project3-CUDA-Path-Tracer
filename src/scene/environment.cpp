// The one TU that compiles tinyexr (v1.0.13); its ZIP decoding uses stb's zlib from stb.cpp, so no miniz.

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_USE_THREAD 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "scene/environment.h"

#include "timing.h"
#include "utilities.h"

#include <stb_image.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

const std::string ENVIRONMENT_DIR = "scenes/assets/hdri/";

namespace
{
void loadEnvironmentImage(const std::string& path, Environment& env)
{
    const std::string ext = lowercaseExtension(path);
    float* pixels = nullptr;
    int width = 0;
    int height = 0;
    if (ext == ".exr")
    {
        // LoadEXR returns RGBA floats, alpha 1 when the file has none.
        const char* err = nullptr;
        if (LoadEXR(&pixels, &width, &height, path.c_str(), &err) != TINYEXR_SUCCESS)
        {
            const std::string reason = err != nullptr ? err : "unknown error";
            FreeEXRErrorMessage(err);
            throw std::runtime_error(path + ": " + reason);
        }
    }
    else if (ext == ".hdr")
    {
        // Radiance only: stbi_loadf would read 8-bit images through an assumed gamma of 2.2.
        int channels = 0;
        pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
        if (pixels == nullptr)
        {
            throw std::runtime_error(path + ": " + stbi_failure_reason());
        }
    }
    else
    {
        throw std::runtime_error(path + ": not an environment map (.hdr or .exr)");
    }

    env.width = width;
    env.height = height;
    env.rgba.assign(pixels, pixels + (size_t)width * height * 4);
    free(pixels);  // both libraries allocate with malloc
    printf("Environment %s: %dx%d\n", path.c_str(), width, height);
}

// Sums run in double: a 4k map has 8M cells, each far below float's resolution next to 1.
void buildEnvironmentTable(Environment& env, bool compensate)
{
    TimingScope timing("load.environment_table");
    const int w = env.width;
    const int h = env.height;
    const size_t cells = (size_t)w * h;

    // A bilinear read averaged over a cell is [1 6 1]/8 per axis; wraps in u, clamps in v, as the texture does.
    std::vector<float> lum(cells);
    for (size_t i = 0; i < cells; ++i)
    {
        const glm::vec3 rgb(env.rgba[4 * i], env.rgba[4 * i + 1], env.rgba[4 * i + 2]);
        lum[i] = std::max(0.0f, luminance(env.radiance * rgb));
    }
    std::vector<float> filteredU(cells);
    for (int j = 0; j < h; ++j)
    {
        const float* row = &lum[(size_t)j * w];
        float* out = &filteredU[(size_t)j * w];
        for (int i = 0; i < w; ++i)
        {
            const int left = i == 0 ? w - 1 : i - 1;
            const int right = i == w - 1 ? 0 : i + 1;
            out[i] = (row[left] + 6.0f * row[i] + row[right]) * (1.0f / 8.0f);
        }
    }
    std::vector<float> filtered(cells);
    for (int j = 0; j < h; ++j)
    {
        const float* up = &filteredU[(size_t)std::max(j - 1, 0) * w];
        const float* mid = &filteredU[(size_t)j * w];
        const float* down = &filteredU[(size_t)std::min(j + 1, h - 1) * w];
        float* out = &filtered[(size_t)j * w];
        for (int i = 0; i < w; ++i)
        {
            out[i] = (up[i] + 6.0f * mid[i] + down[i]) * (1.0f / 8.0f);
        }
    }

    std::vector<double> sinTheta(h);
    for (int j = 0; j < h; ++j)
    {
        sinTheta[j] = std::sin((j + 0.5) * PI / h);
    }

    double integral = 0.0;
    for (int j = 0; j < h; ++j)
    {
        double rowSum = 0.0;
        for (int i = 0; i < w; ++i)
        {
            rowSum += filtered[(size_t)j * w + i];
        }
        integral += rowSum * sinTheta[j];
    }
    integral *= (2.0 * PI / w) * (PI / h);
    env.radianceIntegral = (float)integral;
    const double average = integral / (4.0 * PI);

    // A one-color map compensates to nothing and keeps the plain weights.
    std::vector<double> weight(cells);
    double total = 0.0;
    env.compensated = false;
    if (compensate)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                const size_t c = (size_t)j * w + i;
                weight[c] = std::max(filtered[c] - average, 0.0) * sinTheta[j];
                total += weight[c];
            }
        }
        env.compensated = total > 0.0;
    }
    if (!env.compensated)
    {
        total = 0.0;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                const size_t c = (size_t)j * w + i;
                weight[c] = filtered[c] * sinTheta[j];
                total += weight[c];
            }
        }
    }
    if (total <= 0.0)
    {
        env.radianceIntegral = 0.0f;
        return;
    }

    // A zero-weight row is never picked; its conditional CDF stays uniform so a search in it is defined.
    env.pdfUv.resize(cells);
    env.conditionalCdf.resize(cells);
    env.marginalCdf.resize(h);
    const double mean = total / (double)cells;
    double rowRunning = 0.0;
    for (int j = 0; j < h; ++j)
    {
        double rowSum = 0.0;
        for (int i = 0; i < w; ++i)
        {
            rowSum += weight[(size_t)j * w + i];
        }
        double running = 0.0;
        for (int i = 0; i < w; ++i)
        {
            const size_t c = (size_t)j * w + i;
            env.pdfUv[c] = (float)(weight[c] / mean);
            running += weight[c];
            env.conditionalCdf[c] = rowSum > 0.0 ? (float)(running / rowSum) : (float)(i + 1) / w;
        }
        env.conditionalCdf[(size_t)j * w + w - 1] = 1.0f;
        rowRunning += rowSum;
        env.marginalCdf[j] = (float)(rowRunning / total);
    }
    env.marginalCdf[h - 1] = 1.0f;
    printf("Environment table: %dx%d cells, average luminance %.4g%s\n", w, h, average,
        env.compensated ? ", MIS compensated" : "");
}
}  // namespace

Environment loadEnvironment(const EnvironmentSource& source, bool compensate)
{
    TimingScope timing("load.environment");
    Environment env;
    env.radiance = source.radiance;
    env.rotation = source.rotation;
    if (!source.file.empty())
    {
        loadEnvironmentImage(source.file, env);
        buildEnvironmentTable(env, compensate);
    }
    else
    {
        env.radianceIntegral = 4.0f * PI * luminance(env.radiance);
    }
    return env;
}

std::vector<std::string> environmentMapFiles()
{
    std::vector<std::string> files;
    if (!std::filesystem::is_directory(ENVIRONMENT_DIR))
    {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(ENVIRONMENT_DIR))
    {
        const std::string ext = lowercaseExtension(entry.path().string());
        if (entry.is_regular_file() && (ext == ".hdr" || ext == ".exr"))
        {
            files.push_back(entry.path().generic_string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}
