#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

// Converts linear color format to SRGB
__host__ __device__ inline float linearToSRGB(float c)
{
    c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__host__ __device__ inline glm::vec3 linearToSRGB(glm::vec3 c)
{
    return glm::vec3(linearToSRGB(c.x), linearToSRGB(c.y), linearToSRGB(c.z));
}

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;

    // Runtime toggles
    bool useBVH = true;
    bool sortByMaterial = true;
    bool streamCompaction = true;
    bool visualizeBVH = false;
    int bvhVizMode = 0;          // 0 = leaf colors, 1 = traversal heat map
    bool bvhOutlines = true;
    bool denoise = false;        // Intel Open Image Denoise (only if found at build time)
    bool srgbOutput = true;
    bool useMIS = true;          // explicit light sampling (NEE) + MIS weighting
    bool useDOF = false;
    float fstop = 2.8f;
    float focusDistance = 10.0f; // in world units

    float iterationMs = 0.0f;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
