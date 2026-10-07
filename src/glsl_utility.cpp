// GLSL Utility: A utility class for loading GLSL shaders
// Written by Varun Sampath, Patrick Cozzi, and Karl Li.
// Copyright (c) 2012 University of Pennsylvania

#include "glsl_utility.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

using std::ios;

namespace glsl_utility
{
// embedded passthrough shaders so that default passthrough shaders don't need to be loaded
static std::string passthroughVS =
    "   #version 120 \n"
    "   attribute vec4 Position; \n"
    "   attribute vec2 Texcoords; \n"
    "   varying vec2 v_Texcoords; \n"
    "   \n"
    "   void main(void){ \n"
    "       v_Texcoords = Texcoords; \n"
    "       gl_Position = Position; \n"
    "   }";
static std::string passthroughFS =
    "   #version 120 \n"
    "   varying vec2 v_Texcoords; \n"
    "   \n"
    "   uniform sampler2D u_image; \n"
    "   \n"
    "   void main(void){ \n"
    "       gl_FragColor = texture2D(u_image, v_Texcoords); \n"
    "   }";

typedef struct
{
    GLuint vertex;
    GLuint fragment;
    GLint geometry;
} shaders_t;

char* load_file(const char *fname, GLint &fSize)
{
    // file read based on example in cplusplus.com tutorial
    std::ifstream file (fname, ios::in | ios::binary | ios::ate);
    if (file.is_open())
    {
        unsigned int size = (unsigned int)file.tellg();
        fSize = size;
        char *memblock = new char [size];
        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();
        std::cout << "file " << fname << " loaded" << std::endl;
        return memblock;
    }

    std::cout << "Unable to open file " << fname << std::endl;
    exit(EXIT_FAILURE);
}

// print_shader_info_log
// From OpenGL Shading Language 3rd Edition, p215-216
// Display (hopefully) useful error messages if shader fails to compile
void print_shader_info_log(GLint shader)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1)
    {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        std::cout << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
    }
}

void print_link_info_log(GLint prog)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1)
    {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
        std::cout << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
    }
}

void compile_shader(const char* shaderName, const char * shaderSource, GLenum shaderType, GLint &shaders)
{
    GLint s;
    s = glCreateShader(shaderType);

    GLint slen = (unsigned int)std::strlen(shaderSource);
    char * ss = new char [slen + 1];
    std::strcpy(ss, shaderSource);

    const char * css = ss;
    glShaderSource(s, 1, &css, &slen);

    GLint compiled;
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        std::cout << shaderName << " did not compile" << std::endl;
    }
    print_shader_info_log(s);

    shaders = s;

    delete [] ss;
}

shaders_t load_default_shaders()
{
    shaders_t out;

    compile_shader("Passthrough Vertex", passthroughVS.c_str(), GL_VERTEX_SHADER, (GLint&)out.vertex);
    compile_shader("Passthrough Fragment", passthroughFS.c_str(), GL_FRAGMENT_SHADER, (GLint&)out.fragment);

    return out;
}

shaders_t load_shaders(const char * vert_path, const char * frag_path, const char * geom_path = 0)
{
    shaders_t out;

    // load shaders & get length of each
    GLint vlen, flen, glen;
    char *vertexSource, *fragmentSource, *geometrySource;
    const char *vv, *ff, *gg;

    vertexSource = load_file(vert_path, vlen);
    vv = vertexSource;
    compile_shader("Vertex", vv, GL_VERTEX_SHADER, (GLint&)out.vertex);

    fragmentSource = load_file(frag_path, flen);
    ff = fragmentSource;
    compile_shader("Fragment", ff, GL_FRAGMENT_SHADER, (GLint&)out.fragment);

    if (geom_path)
    {
        geometrySource = load_file(geom_path, glen);
        gg = geometrySource;
        compile_shader("Geometry", gg, GL_GEOMETRY_SHADER, (GLint&)out.geometry);
    }

    return out;
}

void attach_and_link_program( GLuint program, shaders_t shaders)
{
    glAttachShader(program, shaders.vertex);
    glAttachShader(program, shaders.fragment);

    glLinkProgram(program);
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        std::cout << "Program did not link." << std::endl;
    }
    print_link_info_log(program);
}

GLuint create_default_program(const char *attributeLocations[], GLuint numberOfLocations)
{
    glsl_utility::shaders_t shaders = glsl_utility::load_default_shaders();

    GLuint program = glCreateProgram();

    for (GLuint i = 0; i < numberOfLocations; ++i)
    {
        glBindAttribLocation(program, i, attributeLocations[i]);
    }

    glsl_utility::attach_and_link_program(program, shaders);

    return program;
}

GLuint create_program(
    const char *vertexShaderPath, 
    const char *fragmentShaderPath,
    const char *attributeLocations[],
    GLuint numberOfLocations)
{
    glsl_utility::shaders_t shaders = glsl_utility::load_shaders(vertexShaderPath, fragmentShaderPath);

    GLuint program = glCreateProgram();

    for (GLuint i = 0; i < numberOfLocations; ++i)
    {
        glBindAttribLocation(program, i, attributeLocations[i]);
    }

    glsl_utility::attach_and_link_program(program, shaders);

    return program;
}
} // namespace glsl_utility
