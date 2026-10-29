//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include "utilities.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
    glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 rotationMat =   glm::rotate(glm::mat4(1.0f), rotation.x * (float) PI / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(1.0f), rotation.y * (float) PI / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(1.0f), rotation.z * (float) PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), scale);
    return translationMat * rotationMat * scaleMat;
}

std::string lowercaseExtension(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext)
    {
        c = (char)std::tolower((unsigned char)c);
    }
    return ext;
}

void fatal(const char* format, ...)
{
    fflush(stdout);  // keep what was printed before in order
    fprintf(stderr, "error: ");
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

void cudaCheck(cudaError_t result, const char* call, const char* file, int line)
{
    if (result == cudaSuccess)
    {
        return;
    }
    // __FILE__ is a full path, with backslashes on Windows.
    const char* name = file;
    for (const char* c = file; *c != '\0'; ++c)
    {
        if (*c == '/' || *c == '\\')
        {
            name = c + 1;
        }
    }
    fatal("%s failed (%s:%d): %s", call, name, line, cudaGetErrorString(result));
}
