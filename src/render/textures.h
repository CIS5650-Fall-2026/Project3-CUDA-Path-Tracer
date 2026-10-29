#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

class Scene;
struct Environment;

// Scene textures on the GPU: uchar4 arrays with texture objects that decode
// sRGB for color slots. Host code only.

// Device array of texture objects indexed like Scene::textures, or null when the scene has none.
cudaTextureObject_t* texturesInit(const Scene& scene);
void texturesFree();

// The environment as the shade kernel reads it, passed by value.
struct EnvironmentMap
{
    cudaTextureObject_t texture;  // lat-long float4 image; 0 when the environment is one color
    glm::vec3 radiance;           // multiplies the image, or the color itself; zero: no environment
    float cosRotation;
    float sinRotation;
    const float* pdfUv;           // width * height; null when the environment is one color
    const float* conditionalCdf;  // width * height
    const float* marginalCdf;     // height
    int width;
    int height;
};

EnvironmentMap environmentInit(const Environment& env);
void environmentFree();
