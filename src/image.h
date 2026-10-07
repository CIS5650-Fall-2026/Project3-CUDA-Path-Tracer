#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

class Image
{
private:
    int width;
    int height;
    std::vector<glm::vec3> pixels;

public:
    Image(int x, int y);
    ~Image();
    void set_pixel(int x, int y, const glm::vec3 &pixel);
    void save_png(const std::string &base_filename, bool tonemap);
    void save_hdr(const std::string &base_filename);
};
