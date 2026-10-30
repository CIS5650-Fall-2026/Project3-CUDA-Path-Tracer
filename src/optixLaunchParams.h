#pragma once

#include <optix.h>
#include <vector_types.h> //float3 and uint3

// Launch parameters are data supplied for a launch, accessible to its GPU programs.
// Basically a struct of input data that we provide to the GPU programs

//Forward declarations - pointers only need the type names
struct PathSegment;
struct ShadeableIntersection;
struct Material;

//Camera contains GLM types whose constructors trigger dynamic initialization, CUDA rejects this for the global __constant__ LaunchParams params
struct RaygenCamera {
	int width;
	int height;
	float3 position;
	float3 view;
	float3 up;
	float3 right;
	float2 pixelLength;
	float apertureRadius;
	float focalDistance; //different than focal Length
};

// Shared layout used by CPU code and OptiX GPU programs
// The handle identifies the built acceleration structure. It doesn't contain the triangles or the GAS itself, only their GPU allocations
struct LaunchParams {
	OptixTraversableHandle gasHandle;

	// Addresses of existing GPU arrays owned by the CUDA renderer
	const PathSegment* paths;
	ShadeableIntersection* intersections;

	// Number of active paths for this launch
	unsigned int numPaths;

	const float3* vertices;
	const float3* normals;
	const float4* tangents;
	const float2* texcoords;
	const uint3* triangles;
	const int* triangleMaterialIds;

	// Testing raygen loop implementation (everything runs on optix kernel)
	PathSegment* outputPaths; // Write each completed path back for finalGather()
	const Material* materials; //read material colors and emission during shading
	const cudaTextureObject_t* textures;
	cudaTextureObject_t environmentTexture;
	const double* environmentCdf;
	int environmentWidth;
	int environmentHeight;
	int iteration; //seed the random generator for the current sample

	//Also including camera
	RaygenCamera camera;
	int maxBounces;
	int generateCameraRays; // 0 read existing paths, 1 generate in raygen
};