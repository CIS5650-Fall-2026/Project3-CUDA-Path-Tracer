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

#define MAX_STAT_DEPTH 32

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;

    // Toggles shown in the ImGui panel
    bool StreamCompaction = true;
    bool SortByMaterial = false;
    bool AntiAliasing = true;
    bool RussianRoulette = true;
    bool BBoxCulling = true;
    bool DirectLighting = false;
    bool MotionBlur = true;

    // Stats shown in the ImGui panel
    float AvgIterationMs = 0.0f;            // average time per iteration since the last restart
    int AlivePaths[MAX_STAT_DEPTH] = {};    // paths still alive after each bounce, -1 if unknown
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
