// GLSL Utility: A utility class for loading GLSL shaders
// Written by Varun Sampath, Patrick Cozzi, and Karl Li.
// Copyright (c) 2012 University of Pennsylvania

#ifndef GLSL_UTILITY_HPP
#define GLSL_UTILITY_HPP

#include <GL/glew.h>

namespace glsl_utility
{
GLuint create_default_program(const char *attributeLocations[], GLuint numberOfLocations);
GLuint create_program(
    const char *vertexShaderPath,
    const char *fragmentShaderPath,
    const char *attributeLocations[],
    GLuint numberOfLocations);
}

#endif
