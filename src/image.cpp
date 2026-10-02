#include "image.h"

#include <stb_image_write.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

float linearToSrgb(float value)
{
    if (!std::isfinite(value) || value <= 0.0f)
    {
        return 0.0f;
    }

    // A bounded display transform keeps unusually bright Monte Carlo samples
    // representable in an 8-bit preview.  HDR output remains linear.
    const float mapped = value / (1.0f + value); // Reinhard tone mapping.
    return mapped <= 0.0031308f
        ? 12.92f * mapped
        : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
}

} // namespace

Image::Image(int x, int y)
    : xSize(x), ySize(y), pixels(nullptr)
{
    if (x <= 0 || y <= 0)
    {
        throw std::invalid_argument("image dimensions must be positive");
    }

    const std::size_t width = static_cast<std::size_t>(x);
    const std::size_t height = static_cast<std::size_t>(y);
    if (width > std::numeric_limits<std::size_t>::max() / height
        || width * height > std::numeric_limits<std::size_t>::max() / sizeof(glm::vec3)
        || width * height > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || width > static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
    {
        throw std::overflow_error("image dimensions are too large");
    }

    pixels = new glm::vec3[width * height]();
}

Image::~Image()
{
    delete[] pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename)
{
    const std::size_t pixelCount = static_cast<std::size_t>(xSize)
        * static_cast<std::size_t>(ySize);
    std::vector<unsigned char> bytes(3 * pixelCount);
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            const std::size_t i = static_cast<std::size_t>(y) * xSize + x;
            for (int channel = 0; channel < 3; ++channel)
            {
                const float display = glm::clamp(
                    linearToSrgb(pixels[i][channel]), 0.0f, 1.0f);
                bytes[3 * i + channel] = static_cast<unsigned char>(
                    std::lround(display * 255.0f));
            }
        }
    }

    std::string filename = baseFilename + ".png";
    if (stbi_write_png(filename.c_str(), xSize, ySize, 3,
            bytes.data(), xSize * 3) == 0)
    {
        throw std::runtime_error("failed to write PNG image: " + filename);
    }
    std::cout << "Saved " << filename << "." << std::endl;
}

void Image::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    const std::size_t pixelCount = static_cast<std::size_t>(xSize)
        * static_cast<std::size_t>(ySize);
    std::vector<float> values(3 * pixelCount);
    for (std::size_t i = 0; i < pixelCount; ++i)
    {
        values[3 * i + 0] = pixels[i].x;
        values[3 * i + 1] = pixels[i].y;
        values[3 * i + 2] = pixels[i].z;
    }

    if (stbi_write_hdr(filename.c_str(), xSize, ySize, 3, values.data()) == 0)
    {
        throw std::runtime_error("failed to write HDR image: " + filename);
    }
    std::cout << "Saved " + filename + "." << std::endl;
}
