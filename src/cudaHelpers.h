#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) checkCudaResult((call), #call, __FILE__, __LINE__)

inline void checkCudaResult(cudaError_t result, const char* call, const char* file, int line) {
	if (result == cudaSuccess) {
		return;
	}

	std::string message = std::string("CUDA error: ") + cudaGetErrorString(result) +
		"\n  in " + call +
		"\n  at " + file + ":" + std::to_string(line);

	std::cerr << message << std::endl;
	throw std::runtime_error(message);
}

__host__ __device__ inline int divup(int a, int b) {
	return (a + b - 1) / b;
}

// Create enough blocks to cover a 2D grid of the desired size
inline dim3 make2DGrid(int width, int height, const dim3& blockSize) {
	return dim3(divup(width, blockSize.x), divup(height, blockSize.y));
}
