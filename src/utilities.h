#pragma once

#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

// CONTROLLERS 
#define CAMERA_PASS       1
#define PHOTON_PASS       1
#define DISPERSION        1

#define ARTISTIC          1 // visually improves dispersion 
#define SPECULAR_LE_CLAMP 10.0f


#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
inline void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
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
