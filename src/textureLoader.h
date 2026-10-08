#pragma once

#include "stb_image.h"
#include <iostream>
#include <cuda_runtime.h>

class TextureLoader {
public:
	static bool LoadTexture(const char* filepath, cudaTextureObject_t& textureObject, cudaArray_t& textureArray, cudaTextureDesc texDesc);

	// Binary gltf files have the textures in memory so have to use a special helper here
	static bool LoadTextureFromMemory(const unsigned char* encodedBytes, size_t byteCount, cudaTextureObject_t& textureObject, cudaArray_t& textureArray, cudaTextureDesc texDesc);

private:
	// Helper for taking pixel data and uploading it to the gpu
	static bool CreateTextureFromPixels(const unsigned char* data, int width, int height, cudaTextureObject_t& textureObject, cudaArray_t& textureArray, cudaTextureDesc texDesc);
};
