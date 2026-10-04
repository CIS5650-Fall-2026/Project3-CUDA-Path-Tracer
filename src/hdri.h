#pragma once

#include <string>
#include <vector>

// loads an equirectangular HDR environment map as RGBA floats.
// Starts with top row. .exr file is read with tinyexr, .hdr is read with stb_image.
// Returns whether or not succeeded, prints errors on failure.
bool loadHDRI(const std::string& path, std::vector<float>& rgba, int& width, int& height);
