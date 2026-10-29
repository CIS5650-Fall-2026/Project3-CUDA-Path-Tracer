#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

// A lat-long file times radiance, or radiance alone; zero radiance means no environment.
struct EnvironmentSource
{
    std::string file;  // a .hdr or .exr; empty for one color
    glm::vec3 radiance = glm::vec3(0.0f);
    // Radians about +Y applied to the lookup direction, as a Blender Mapping node's Z rotation.
    float rotation = 0.0f;
};

// Lat-long, rows top to bottom: u = 0.5 + atan2(d.z, d.x) / (2 pi), v = acos(d.y) / pi.
// The table follows luminance * sin(theta) of the bilinear read averaged per cell; MIS compensation per Karlik et al. 2019.
struct Environment
{
    glm::vec3 radiance = glm::vec3(0.0f);  // multiplies the image, or the color itself
    float rotation = 0.0f;
    int width = 0;
    int height = 0;
    std::vector<float> rgba;               // width * height * 4; empty when there is no image

    // Luminance integrated over the sphere, W/m^2.
    float radianceIntegral = 0.0f;
    // Empty without an image. pdfUv has mean 1 over the cells; over directions it is pdfUv / (2 pi^2 sin(theta)).
    std::vector<float> pdfUv;           // width * height
    std::vector<float> conditionalCdf;  // width * height
    std::vector<float> marginalCdf;     // height
    bool compensated = false;           // the table had the average subtracted
};

// Throws std::runtime_error naming the file when it cannot be read.
Environment loadEnvironment(const EnvironmentSource& source, bool compensate);

extern const std::string ENVIRONMENT_DIR;
std::vector<std::string> environmentMapFiles();
