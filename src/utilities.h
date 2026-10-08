#pragma once

#include "glm/glm.hpp"

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
#define IPI               0.31830988618f
#define BOUNCE_EPS 0.005f

#define SQR(x) ((x)*(x))

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;

    // Settings
    bool SortRaysByMaterial;
    int FrameIndex = 0;
    int AnimationIterationCount = 1000;
    
    int MaxDepth = 5;
    int RenderType = 1;
    bool UseDepthOfField = false;
    float FocalDistance = 11.5f;
    float ApertureFNumber = 10.0f;
    int ColorCorrection = 1;

    float DensityMultiplier = 1.0f;
    float TemperatureScale = 1.0f;
    float EmissionScale = 50.0f;
    float EmissionContrast = 2.0f;

    bool UseTemperature = true;
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