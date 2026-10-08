#include "textureLoader.h"

bool TextureLoader::LoadTexture(const char* filepath, cudaTextureObject_t& textureObject, cudaArray_t& textureArray, cudaTextureDesc texDesc) {
	int width, height, channels;
	unsigned char* data = stbi_load(filepath, &width, &height, &channels, 4);

	if (!data) {
		std::cerr << "Failed to load texture: " << filepath << std::endl;
		return false;
	}

	bool success = CreateTextureFromPixels(data, width, height, textureObject, textureArray, texDesc);
	stbi_image_free(data);
	return success;
}

// Binary gltf files have the textures in memory so have to use a special helper here
bool TextureLoader::LoadTextureFromMemory(const unsigned char* encodedBytes, size_t byteCount, cudaTextureObject_t& textureObject, cudaArray_t& textureArray, cudaTextureDesc texDesc) {
	int width, height, channels;
	unsigned char* data = stbi_load_from_memory(encodedBytes, static_cast<int>(byteCount), &width, &height, &channels, 4);

	if (!data) {
		std::cerr << "Failed to decode embedded texture: " << stbi_failure_reason() << std::endl;
		return false;
	}

	bool success = CreateTextureFromPixels(data, width, height, textureObject, textureArray, texDesc);
	stbi_image_free(data);
	return success;
}

bool TextureLoader::CreateTextureFromPixels(const unsigned char* data, int width, int height, cudaTextureObject_t& textureObject, cudaArray_t& textureArray, cudaTextureDesc texDesc) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaError_t err = cudaMallocArray(&textureArray, &channelDesc, width, height);
	if (err != cudaSuccess) {
		std::cerr << "[CUDA] Failed to allocate array: " << cudaGetErrorString(err) << std::endl;
		return false;
	}

	size_t pitch = width * 4 * sizeof(unsigned char);
	cudaError_t memcpyResult = cudaMemcpy2DToArray(
		textureArray,
		0, 0,
		data,
		pitch,
		pitch,
		height,
		cudaMemcpyHostToDevice
	);
	if (memcpyResult != cudaSuccess) {
		std::cerr << "[CUDA] Failed to copy texture data to array: " << cudaGetErrorString(memcpyResult) << std::endl;
		cudaFreeArray(textureArray);
		textureArray = nullptr;
		return false;
	}

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = textureArray;

	err = cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr);
	if (err != cudaSuccess) {
		std::cerr << "[CUDA] Failed to create texture object: " << cudaGetErrorString(err) << std::endl;
		cudaFreeArray(textureArray);
		textureArray = nullptr;
		return false;
	}

	return true;
}