#include "image.h"
#include "display.cuh"

#include <stb_image_write.h>

#include <iostream>
#include <string>


Image::Image(int x, int y) : width(x), height(y), pixels(x * y) {}

Image::~Image() {}

void Image::set_pixel(int x, int y, const glm::vec3& pixel) {
    assert(x >= 0 && y >= 0 && x < width && y < height);
    pixels[(y * width) + x] = pixel;
}

void Image::save_png(const std::string& base_filename, bool tonemap) {
    unsigned char* bytes = new unsigned char[3 * width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            glm::vec3 pix = to_display(pixels[i], tonemap);
            bytes[3 * i + 0] = (unsigned char)pix.x;
            bytes[3 * i + 1] = (unsigned char)pix.y;
            bytes[3 * i + 2] = (unsigned char)pix.z;
        }
    }

    std::string filename = base_filename + ".png";
    std::string file_path = filename;
    stbi_write_png(file_path.c_str(), width, height, 3, bytes, width * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::save_hdr(const std::string& base_filename) {
    std::string filename = base_filename + ".hdr";
    stbi_write_hdr(filename.c_str(), width, height, 3, (const float*)pixels.data());
    std::cout << "Saved " + filename + "." << std::endl;
}
