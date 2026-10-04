#include "hdri.h"

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <stb_image.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>

bool loadHDRI(const std::string& path, std::vector<float>& rgba, int& width, int& height)
{
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });

    if (ext == "exr")
    {
        // LoadEXR converts half/uint channels to float and always returns 4 channels (alpha = 1 if missing)
        float* data = nullptr;
        const char* err = nullptr;
        if (LoadEXR(&data, &width, &height, path.c_str(), &err) != TINYEXR_SUCCESS)
        {
            std::cout << "error! tinyexr couldn't load " << path << ": " << (err ? err : "unknown error") << std::endl;
            if (err) FreeEXRErrorMessage(err);
            return false;
        }
        rgba.assign(data, data + (size_t)width * height * 4);
        free(data);
        return true;
    }

    if (ext == "hdr")
    {
        int channels;
        float* data = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
        if (!data)
        {
            std::cout << "error! stb_image couldn't load " << path << ": " << stbi_failure_reason() << std::endl;
            return false;
        }
        rgba.assign(data, data + (size_t)width * height * 4);
        stbi_image_free(data);
        return true;
    }

    std::cout << "error! Unsupported HDRI format (use .exr or .hdr): " << path << std::endl;
    return false;
}
