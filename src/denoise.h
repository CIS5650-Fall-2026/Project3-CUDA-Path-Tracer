#pragma once

// Wrapper for Intel Open Image Denoise (OIDN). When CMake can't find OIDN, USE_OIDN is not
// defined, and denoiser available defaults to false.

// True when denoiser has been initialized (needs OIDN at build time and a supported NVIDIA GPU).
bool denoiserAvailable();

// color, albedo, normal and output are cudaMalloc'ed float3 images of width * height pixels.
// They must stay allocated until denoiserFree(). Returns false if the denoiser can't be used.
bool denoiserInit(int width, int height, float* color, float* albedo, float* normal, float* output);

// Denoises albedo and normal into output. Blocks until finished.
void denoiserRun();

void denoiserFree();
