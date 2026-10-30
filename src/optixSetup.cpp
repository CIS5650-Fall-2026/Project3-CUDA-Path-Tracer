#ifndef NOMINMAX
#define NOMINMAX
#endif

// initialization and cleanup function declarations
#include "optixSetup.h"

// CUDA Runtime API, including cudaFree()
#include <cuda_runtime.h>

// Optix APi types, including OptixDeviceContext
#include <optix.h>

// Defines the function table that holds OptiX function pointers
// Must be inluded in exactly ONE .cpp file within the application
#include <optix_function_table_definition.h>

// Provides optixInit() and wrapperes that call through that table
#include <optix_stubs.h>

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>

#include "optixLaunchParams.h"

#include <vector>
#include "gltfLoader.h"

#include <optix_stack_size.h>
#include "sceneStructs.h"

#define TEST_LAUNCH 0

//PTX: Intermediate GPu instructions generated from optixPrograms.cu - generated on build, not runtime (created by specifying __raygen__rg and tweaking cmakeslists)
//Module: OPtiX compiles the PTX into a module containing the GPU program
//Program group: Selects __raygen__rg from that module as the raygen program
//Pipeline: Links the selected program groups into an exectuable GPU pipeline
//SBT: Supplies records identifying which programs to invoke for a particular launch, plus optional data - it's like binding shaders to a renderer

//optixContext: Create context - identifies the shared OptiX environment where we create modules, program groups, and pipelines
//optixModule: Create module from PTX - Identifies the module containing the compiled GPU program
//pipelineCompileOptions: Configure Compilation - Stores settings shared by module and pipeline creation. Settings, not a handle
//raygenProgramGroup: Select raygen function - Identifies the program group selection __raygen__rg from optixModule
//optixPipeline: Link pipeline - identifies the executable pipeline built from that program group
//dev_raygenRecord: Upload SBT record - Points to the GPU allocation containing the packed raygen record
//sbt: Describe SBT + Launch: CPU-side structure whose raygenRecord field holds that GPU address, will be passed into launch call

namespace { // Anonymous namespace makes names private to this .cpp file
	// Handle to the OptiX device context
	// nullptr means we have not created a context yet
	OptixDeviceContext optixContext = nullptr;

	// Handle to the module containing our GPU program - moduleOptiosn controls compilation choices such as optimization
	OptixModule optixModule = nullptr;

	// Keeping settings so we can resue them when creating the pipeline
	// Module creation and pipeline creation must use consistent settings - describes features the eventual pipeline will use (motion blur, payload values, attribute values etc)
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	// optixContext - holds optiX state associated with the CUDA context. Modules and pipelines belong to it
	// optixModule - contains compiled GPU programs from the PTX (__raygen__rg())
	// pipelineCompileOptions - a settings structure describing pipeline features. isn't a context or an executeable pipeline

	// Handle to the program group selecting the raygen function - tells OptiX to use the function named __raygen__rg from this module as a ray generation program
	// It's called a group, but a raygen program grou pselects just one entry function - a hit group can combine closest-hit, any-hit, and intersection functions
	// Selects functions from a module and specifies their roles in the pipeline
	OptixProgramGroup raygenProgramGroup = nullptr;

	// Miss program group - tells OptiX which program to run when a traced ray hits no geometry
	OptixProgramGroup missProgramGroup = nullptr;

	// Handle to the executable pipeline that links the selected GPU programs, creating it does not launch GPU work
	OptixPipeline optixPipeline = nullptr;
	
	// Selects the programs used when a ray intersects our triangle
	OptixProgramGroup hitgroupProgramGroup = nullptr;

	// Temporary GPU workspace used during GAS construction
	void* dev_gasTempBuffer = nullptr;
	// GPU allocation holding the completed acceleration structure
	void* dev_gasOutputBuffer = nullptr;
	// Opaque identifier returned by Optix for the completed GAS
	OptixTraversableHandle gasHandle = 0;

	// GPU buffer holding the launch parameters supplied to OptixLaunch()
	LaunchParams* dev_launchParams = nullptr;

	//CPU vector holding handles to GPU image allocations
	std::vector<cudaArray_t> dev_imageArrays;

	//one CUDA texture object pre TeextureData entry, conains texture handles
	std::vector<cudaTextureObject_t> textureObjects;

	//CPU vector of texture handles
	cudaTextureObject_t* dev_textureObjects = nullptr;

	//dev_enviornmentARray stores pixels fo texture, while environmentTexture provides a way to sample those textures
	cudaArray_t dev_environmentArray = nullptr;
	cudaTextureObject_t environmentTexture = 0;

	double* dev_environmentCdf = nullptr;
	int environmentWidth = 0;
	int environmentHeight = 0;

	// Read the generated PTX file into CPU memory
	std::string loadPtxFile(const char* path) {
		std::ifstream file(path, std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error(std::string("Could not open PTX file: ") + path);
		}

		std::string ptx{
			std::istreambuf_iterator<char>(file),
			std::istreambuf_iterator<char>()
		};

		if (ptx.empty()) {
			throw std::runtime_error(std::string("PTX file is empty: ") + path);
		}

		return ptx;
	}

	//////////////////////////////////
	// Shader binding table (SBT) - provides records identifying the programs to use during a launch, along wiht optional program-specific data
	//////////////////////////////////

	// Define the layout for the raygen SBT record
	// alignas: ensures that records satisfy OptiX's alignment requirement
	struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { //identifies which OptiX program group to invoke
		// OptiX will fill this with information identifying the program
		char header[OPTIX_SBT_RECORD_HEADER_SIZE]; //binary storage, not a text string - OptiX defines it as 32 bytes and requires SBT record alignment of 16 bytes
	};

	//RaygenLoop Implementation
	OptixProgramGroup pathtraceRaygenProgramGroup = nullptr;
	OptixPipeline pathtracePipeline = nullptr;
	RaygenRecord* dev_pathtraceRaygenRecord = nullptr;
	OptixShaderBindingTable pathtraceSbt = {};

	// Same header only layout, it will identify our hitgroup
	using HitgroupRecord = RaygenRecord;

	// GPU allocation containing the packed record
	HitgroupRecord* dev_hitgroupRecord = nullptr;

	//Header only layout - same layout as our raygen record
	using MissRecord = RaygenRecord;
	MissRecord* dev_missRecord = nullptr;

	// Pointer to teh GPU allocation that holds the raygen record, no GPU memory memory actually allocated
	RaygenRecord* dev_raygenRecord = nullptr;

	// CPU-side description of where the SBT records are stored on the GPU, zero initialized
	OptixShaderBindingTable sbt = {};

	// Detailed diagnostics reported by OptiX to the CPU
	void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
		std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << std::endl;
	}

	//GPU allocation containing the three vertices of a test triangle
	float3* dev_meshVertices = nullptr;
	float3* dev_meshNormals = nullptr;
	float4* dev_meshTangents = nullptr;
	float2* dev_meshTexcoords = nullptr;
	uint3* dev_meshIndices = nullptr;
	int* dev_triangleMaterialIds = nullptr;

	//Error printing Helpers
	void checkCuda(cudaError_t result, const char* operation) {
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
		}
	}

	void checkOptix(OptixResult result, const char* operation) {
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string(operation) + ": " + optixGetErrorString(result));
		}
	}

	//PathTracing on optix
	//Same as initOptixContext
	void initOptixPathtracePipeline()
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; //raygen program
		desc.raygen.module = optixModule;
		desc.raygen.entryFunctionName = "__raygen__pathtrace";

		OptixProgramGroupOptions options = {};

		OptixResult optixResult = optixProgramGroupCreate(
			optixContext,
			&desc,
			1,
			&options,
			nullptr,
			nullptr,
			&pathtraceRaygenProgramGroup);
		checkOptix(optixResult, "Create pathtrace raygen program group");

		// Reuse the existing hit and miss, only need new raygen
		OptixProgramGroup groups[] = {
			pathtraceRaygenProgramGroup,
			missProgramGroup,
			hitgroupProgramGroup
		};

		OptixPipelineLinkOptions linkOptions = {};

		// Sequential trace calls, not recursive trace calls
		linkOptions.maxTraceDepth = 1;

		optixResult = optixPipelineCreate(
			optixContext,
			&pipelineCompileOptions,
			&linkOptions,
			groups,
			3,
			nullptr,
			nullptr,
			&pathtracePipeline);
		checkOptix(optixResult, "Create pathtrace pipeline");

		// Compute stack requirements for raygen loop program
		OptixStackSizes stackSizes = {};

		for (OptixProgramGroup group : groups) {
			optixResult = optixUtilAccumulateStackSizes(
				group,
				&stackSizes,
				pathtracePipeline);
			checkOptix(optixResult, "Accumulate pathtrace stack sizes");
		}

		unsigned int directFromTraversal = 0;
		unsigned int directFromState = 0;
		unsigned int continuation = 0;

		optixResult = optixUtilComputeStackSizes(
			&stackSizes,
			1,  // Maximum nested optixTrace depth
			0,  // No continuation callables
			0,  // No direct callables
			&directFromTraversal,
			&directFromState,
			&continuation);
		checkOptix(optixResult, "Compute pathtrace stack sizes");

		optixResult = optixPipelineSetStackSize(
			pathtracePipeline,
			directFromTraversal,
			directFromState,
			continuation,
			1); // Traversable depth: a single GAS.
		checkOptix(optixResult, "Set pathtrace pipeline stack sizes");

		RaygenRecord record = {};

		optixResult = optixSbtRecordPackHeader(
			pathtraceRaygenProgramGroup,
			&record);
		checkOptix(optixResult, "Pack pathtrace raygen record");

		cudaError_t cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_pathtraceRaygenRecord), sizeof(RaygenRecord));
		checkCuda(cudaResult, "Allocate pathtrace raygen record");

		cudaResult = cudaMemcpy(dev_pathtraceRaygenRecord, &record, sizeof(RaygenRecord), cudaMemcpyHostToDevice);
		checkCuda(cudaResult, "Upload pathtrace raygen record");

		// Share the existing hit/miss records, select the new raygen
		pathtraceSbt = sbt;
		pathtraceSbt.raygenRecord = reinterpret_cast<CUdeviceptr>(dev_pathtraceRaygenRecord);
	}

	cudaTextureAddressMode getTextureAddressMode(int wrapMode) {
		switch(wrapMode) {
		case 10497: //REPEAT
			return cudaAddressModeWrap;

		case 33071: //CLAMP_TO_EDGE
			return cudaAddressModeClamp;

		case 33648: //MIRRORED_REPEAT
			return cudaAddressModeMirror;

		default:
			throw std::runtime_error("Unsupported texture wrap mode");
		}
	}

}

void initOptixContext(const std::vector<MeshData>& meshes, const std::vector<ImageData>& images, const std::vector<TextureData>& textures, const EnvironmentData& environment, int lightMaterialId) {
	// Ensure Cuda is initialized for the current device, passing nullptr frees no allocation
	// Stop initialization if CUDA reports an error
	cudaError_t cudaResult = cudaFree(nullptr);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA initialization failed: ") + cudaGetErrorString(cudaResult));
	}

	//////////////////////////////////
	// Load the OptiX driver API and populate its function table
	//////////////////////////////////
	OptixResult optixResult = optixInit();
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("Optix initialization failed. Eror code: ") + std::to_string(static_cast<int>(optixResult)));
	}

	//////////////////////////////////
	// Create Optix Context
	//////////////////////////////////
	// Zero-initialize all options, these options configure the context we'll create
	OptixDeviceContextOptions options = {};

	// Enable diagnostic messages, including detailed information
	options.logCallbackFunction = optixLogCallback;
	options.logCallbackLevel = 4;

	// Enable additional checks while debugging the setup
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;

	// nullptr tells OptiX to use the current CUDA context, configured with &options, and writes the created handle into &optixContext.
	// reusing existing optixREsult 
	optixResult = optixDeviceContextCreate(nullptr, &options, &optixContext);

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX context creation failed: ") + optixGetErrorString(optixResult));
	}

	std::cout << "OptiX context created." << std::endl;

	// Print the PTX path supplied by CMake
	std::cout << "OptiX PTX path: " << OPTIX_PTX_PATH << std::endl;

	// Load the intermediate GPU program using the path supplied by CMake
	std::string ptx = loadPtxFile(OPTIX_PTX_PATH);

	// Confirm we actually read the file
	std::cout << "Loaded PTX: " << ptx.size() << " bytes." << std::endl;

	// Configure compilation of this module
	OptixModuleCompileOptions moduleOptions = {};
	moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	// First test only runs raygen, does not trace rays
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.numPayloadValues = 0;
	pipelineCompileOptions.numAttributeValues = 2; // built-in triangle intersections provide 2 barycentric coordinates
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // test pipeline uses triangle geometry

	// Haven't declared a GPU laun-parameter variable yet
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

	// OptiX writes compiler diagnostics into this CPU buffer
	char log[4096] = {};
	size_t logSize = sizeof(log);

	//////////////////////////////////
	// Create Optix Module
	//////////////////////////////////
	optixResult = optixModuleCreate(optixContext, &moduleOptions, &pipelineCompileOptions, 
		ptx.c_str(), // Pointer to PTX text
		ptx.size(), // Length of the text in bytes
		log, 
		&logSize, // Buffer capacity in, log size out
		&optixModule);
	
	// Ensure printing stays within the buffer even if the log was truncated
	log[sizeof(log) - 1] = '\0';
	if (log[0] != '\0') {
		std::cout << "OptiX module log:\n" << log << std::endl;
	}

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX module creation failed: ") + optixGetErrorString(optixResult));
	}

	std::cout << "Optix module created." << std::endl;

	// Describe which GPU function we want and its role
	OptixProgramGroupDesc raygenDesc = {};
	raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; //kind is raygen
	raygenDesc.raygen.module = optixModule;
	raygenDesc.raygen.entryFunctionName = "__raygen__rg";

	// Default program-group options are sufficient for initial test
	OptixProgramGroupOptions programGroupOptions = {};

	// Reuse the diagnostic buffer. reset its capacity because the previous call changed logSize
	log[0] = '\0';
	logSize = sizeof(log);

	//////////////////////////////////
	// Create raygen program group - selects functions from a module and specifies their roles in the pipeline
	//////////////////////////////////

	optixResult = optixProgramGroupCreate(
		optixContext, 
		&raygenDesc,
		1,// Number of program groups to create
		&programGroupOptions,
		log,
		&logSize,
		&raygenProgramGroup); //receives the created handle

	log[sizeof(log) - 1] = '\0';
	if (log[0] != '\0') {
		std::cout << "OptiX program group log:\n" << log << std::endl;
	}

	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX raygen program group creation failed: ") + optixGetErrorString(optixResult));
	}

	std::cout << "OptiX raygen program group created." << std::endl;

	//////////////////////////////////
	// Create miss program group
	//////////////////////////////////
	OptixProgramGroupDesc missDesc = {};
	missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	//Replace the empty miss behavior with the GPU function
	missDesc.miss.module = optixModule;
	missDesc.miss.entryFunctionName = "__miss__ms";

	optixResult = optixProgramGroupCreate(optixContext,
		&missDesc,
		1,
		&programGroupOptions,
		nullptr,
		nullptr,
		&missProgramGroup);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("Miss program group creation failed: ")+ optixGetErrorString(optixResult));
	}


	//////////////////////////////////
	// Create hit program group
	//////////////////////////////////
	OptixProgramGroupDesc hitgroupDesc = {};
	hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	//CH = closest-hit
	hitgroupDesc.hitgroup.moduleCH = optixModule;
	hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

	//AH remains null: we aren't using an any-hit program
	//IS remains null: triangles use Optix's built in intersection

	optixResult = optixProgramGroupCreate(
		optixContext,
		&hitgroupDesc,
		1,
		&programGroupOptions,
		nullptr, // the context callback supplies diagnostics
		nullptr,
		&hitgroupProgramGroup
	);
	checkOptix(optixResult, "Hitgroup program group creation failed");

	std::cout << "OptiX hitgroup program group created." << std::endl;

	//////////////////////////////////
	// Link raygen program group into pipeline
	//////////////////////////////////
		// Pipeline includes these
	OptixProgramGroup programGroups[] = {
		raygenProgramGroup,
		missProgramGroup,
		hitgroupProgramGroup
	};

	OptixPipelineLinkOptions linkOptions = {};

	// Trace recursion depth, not the rendereer's total bounce count
	linkOptions.maxTraceDepth = 1;

	log[0] = '\0';
	logSize = sizeof(log);

	optixResult = optixPipelineCreate(
		optixContext, // Context that owns the OptiX objects
		&pipelineCompileOptions, // Same compile settings for the module
		&linkOptions, // Settings for linking the pipeline
		programGroups, // Address for our single program-group handle
		3, // Number of program groups
		log, // diagnostics
		&logSize, 
		&optixPipeline // handle
	);

	log[sizeof(log) - 1] = '\0';
	if (log[0] != '\0') {
		std::cout << "OptiX pipeline log: \n" << log << std::endl;
	}
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX pipeline creation failed: ") + optixGetErrorString(optixResult));
	}
	std::cout << "OptiX pipeline created." << std::endl;

	//////////////////////////////////
	// Create a record in CPU memory
	//////////////////////////////////
	RaygenRecord raygenRecord = {}; //initially filled with zeros


	// Ask OptiX to write the program group's identifying information into the record's header
	optixResult = optixSbtRecordPackHeader(
		raygenProgramGroup, // Program group that the record will identify
		&raygenRecord // Address of the CPU record to fill
	);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX raygen record header packing failed: ") + optixGetErrorString(optixResult));
	}
	std::cout << "OptiX raygen record header packed." << std::endl; //Packing: writing OptiX's internal binary information into the header

	//////////////////////////////////
	// Copy packed raygen record to GPU memory
	//////////////////////////////////
	
	// Allocate GPU memory for one raygen record, dev_raygenRecord has the allocation's address
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_raygenRecord), sizeof(RaygenRecord));
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Raygen record allocation failed: ") + cudaGetErrorString(cudaResult));
	}

	// Copy the packed CPU record into the GPU allocation
	cudaResult = cudaMemcpy(
		dev_raygenRecord,
		&raygenRecord, // Source is our local GPU record
		sizeof(RaygenRecord),
		cudaMemcpyHostToDevice
	);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Raygen record upload failed: ") + cudaGetErrorString(cudaResult));
	}

	// Tell the SBT description where the GPU record lives, CUdeviceptr is the device-address type expected by OptiX. Converts the address representation, copies no data
	sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(dev_raygenRecord);

	std::cout << "OptiX raygen SBT record uploaded." << std::endl;

	//////////////////////////////////
	// Miss record stuff
	//////////////////////////////////

	// Prepare the miss record in CPU memory
	MissRecord missRecord = {};
	optixResult = optixSbtRecordPackHeader(
		missProgramGroup, &missRecord
	);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("Miss record packing failed: ") + optixGetErrorString(optixResult));
	}

	//Allocate GPU storage for this record
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_missRecord), sizeof(MissRecord));
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Miss record allocation failed: ") + cudaGetErrorString(cudaResult));
	}

	//Upload packed record
	cudaResult = cudaMemcpy(dev_missRecord, &missRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("Miss record upload failed: ") + cudaGetErrorString(cudaResult));
	}

	//Describe the array of miss records: address, spacing, and count
	sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(dev_missRecord);
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = 1;

	//////////////////////////////////
	// Hit record stuff
	//////////////////////////////////
	HitgroupRecord hitgroupRecord = {};
	optixResult = optixSbtRecordPackHeader(hitgroupProgramGroup, &hitgroupRecord);
	checkOptix(optixResult, "Hitgroup record packing failed");

	//Allocate storage for one record on the GPU
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_hitgroupRecord), sizeof(HitgroupRecord));
	checkCuda(cudaResult, "Hitgroup record allocation failed");

	//Copy the packed header into GPU memory
	cudaResult = cudaMemcpy(dev_hitgroupRecord, &hitgroupRecord, sizeof(HitgroupRecord), cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Hitgroup record upload failed");

	//Describe the GPU record array to Optix
	sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(dev_hitgroupRecord);
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = 1;

	//////////////////////////////////
	// HDR Load
	//////////////////////////////////
	if (environment.width <= 0 || environment.height <= 0) {
		throw std::runtime_error("Invalid environment dimensions");
	}

	const size_t envWidth = static_cast<size_t>(environment.width);
	const size_t envHeight = static_cast<size_t>(environment.height);
	const size_t envRowBytes = envWidth * 4 * sizeof(float);

	if (environment.pixels.size() != envWidth * envHeight * 4) {
		throw std::runtime_error("Invalid environment pixel count");
	}

	const cudaChannelFormatDesc envChannelDesc = cudaCreateChannelDesc<float4>();

	cudaResult = cudaMallocArray(&dev_environmentArray, &envChannelDesc, envWidth, envHeight);
	checkCuda(cudaResult, "Environment array allocation failed");

	cudaResult = cudaMemcpy2DToArray(dev_environmentArray, 0, 0, environment.pixels.data(), envRowBytes, envRowBytes, envHeight, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Environment upload failed");

	cudaResourceDesc envResourceDesc = {};
	envResourceDesc.resType = cudaResourceTypeArray;
	envResourceDesc.res.array.array = dev_environmentArray;

	cudaTextureDesc envTextureDesc = {};
	envTextureDesc.addressMode[0] = cudaAddressModeWrap; //horizontal coordinate wraps around the panorama
	envTextureDesc.addressMode[1] = cudaAddressModeClamp; //vertical coordinate clamps at the poles
	envTextureDesc.normalizedCoords = 1;
	envTextureDesc.filterMode = cudaFilterModeLinear;
	envTextureDesc.readMode = cudaReadModeElementType;
	envTextureDesc.sRGB = 0; // float 4 preserves HDR, sRGB = 0 avoids decoding early linear radiance

	cudaResult = cudaCreateTextureObject(&environmentTexture, &envResourceDesc, &envTextureDesc, nullptr);
	checkCuda(cudaResult, "Environment texture creation failed");

	std::cout << "Uploaded HDR environment and created texture object\n";

	//////////////////////////////////
	// CDF Load for environment map BSDF
	//////////////////////////////////
	size_t envPixelCount = static_cast<size_t>(environment.width) * environment.height;
	if (environment.cdf.size() != envPixelCount + 1) {
		throw std::runtime_error("Invalid environment CDF size");
	}

	size_t cdfBytes = environment.cdf.size() * sizeof(double);
	
	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_environmentCdf), cdfBytes);
	checkCuda(cudaResult, "Environment CDF allocation failed");

	cudaResult = cudaMemcpy(dev_environmentCdf, environment.cdf.data(), cdfBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Environment CDF upload failed");

	environmentWidth = environment.width;
	environmentHeight = environment.height;

	std::cout << "Uploaded environment CDF\n";


	//////////////////////////////////
	// Image Load
	//////////////////////////////////
	dev_imageArrays.resize(images.size(), nullptr);

	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	for (size_t i = 0; i < images.size(); i++) {
		const ImageData& image = images[i];

		if (image.width <= 0 || image.height <= 0) {
			throw std::runtime_error("Invalid image dimensions");
		}
		if (image.bitsPerChannel != 8 || image.channels != 4) {
			throw std::runtime_error("Texture upload currently requires 8-bit RGBA images");
		}

		const size_t width = static_cast<size_t>(image.width);
		const size_t height = static_cast<size_t>(image.height);
		const size_t rowBytes = width * sizeof(uchar4);

		if (image.pixels.size() / rowBytes != height || image.pixels.size() % rowBytes != 0) {
			throw std::runtime_error("Image pixel data size is invalid");
		}

		cudaResult = cudaMallocArray(&dev_imageArrays[i], &channelDesc, width, height);
		checkCuda(cudaResult, "Image array allocation failed");

		cudaResult = cudaMemcpy2DToArray(dev_imageArrays[i], 0, 0, image.pixels.data(), rowBytes, rowBytes, height, cudaMemcpyHostToDevice);
		checkCuda(cudaResult, "Image upload failed");
	}

	std::cout << "Uploaded " << dev_imageArrays.size() << "image(s)\n";

	//////////////////////////////////
	// Texture Objects
	//////////////////////////////////
	//Two handles per gltf texture
	//[2*i] = color sampling, with sRGB decoding
	//[2*i+1] = data sampling, without sRGB decoding
	textureObjects.resize(2 * textures.size(), 0);

	for (size_t i = 0; i < textures.size(); i++) {
		const TextureData& texture = textures[i];

		if (texture.imageIndex < 0 || size_t(texture.imageIndex) >= dev_imageArrays.size()) {
			throw std::runtime_error("Invalid texture image index");
		}

		//Both handles reference the same image allocation
		cudaResourceDesc resourceDesc = {};
		resourceDesc.resType = cudaResourceTypeArray;
		resourceDesc.res.array.array = dev_imageArrays[texture.imageIndex];

		cudaTextureDesc textureDesc = {};
		textureDesc.addressMode[0] = getTextureAddressMode(texture.wrapS);
		textureDesc.addressMode[1] = getTextureAddressMode(texture.wrapT);

		textureDesc.normalizedCoords = 1;
		textureDesc.filterMode = cudaFilterModeLinear;
		textureDesc.readMode = cudaReadModeNormalizedFloat;

		//Base color - convert RGB values to linear
		textureDesc.sRGB = 1;

		cudaResult = cudaCreateTextureObject(&textureObjects[2 * i], &resourceDesc, &textureDesc, nullptr);
		checkCuda(cudaResult, "Color texture object creation failed");

		//Metallic/roughness and normals - retain linear data values
		textureDesc.sRGB = 0;

		cudaResult = cudaCreateTextureObject(&textureObjects[2 * i + 1], &resourceDesc, &textureDesc, nullptr);
		checkCuda(cudaResult, "Data texture object creation failed");
	}
	std::cout << "Created " << textureObjects.size() << " texture object(s)\n";

	if (!textureObjects.empty()) {
		const size_t textureBytes = textureObjects.size() * sizeof(cudaTextureObject_t);

		cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_textureObjects), textureBytes);
		checkCuda(cudaResult, "Texture handle allocation failed");

		cudaResult = cudaMemcpy(dev_textureObjects, textureObjects.data(), textureBytes, cudaMemcpyHostToDevice);
		checkCuda(cudaResult, "Texture handle upload failed");
	}

	//////////////////////////////////
	// Triangle Load
	//////////////////////////////////
	if (meshes.empty()) {
		throw std::runtime_error("No mesh parts loaded");
	}

	//combine data from all separate meshes into 1 array
	std::vector<float3> vertices;
	std::vector<float3> vertexNormals;
	std::vector<float2> vertexTexcoords;
	std::vector<float4> vertexTangents;
	std::vector<uint3> triangles;
	std::vector<int> triangleMaterialIds;

	//for each mesh
	for (const MeshData& mesh : meshes) {
		if (mesh.positions.empty() || mesh.triangles.empty()) {
			throw std::runtime_error("Mesh geometry is empty");
		}
		if (mesh.rendererMaterialId < 0) {
			throw std::runtime_error("Mesh has no renderer material assigned");
		}

		//Where the mesh's vertices begin in the combined buffer
		const unsigned int vertexOffset = (unsigned int)vertices.size();

		for (const glm::vec3& p : mesh.positions) {
			vertices.push_back(make_float3(p.x, p.y, p.z));
		}

		if (!mesh.normals.empty() && mesh.normals.size() != mesh.positions.size()) {
			throw std::runtime_error("Normal count does not match vertex count");
		}

		for (size_t v = 0; v < mesh.positions.size(); ++v) {
			if (mesh.normals.empty()) {
				// fallback for 0s will be implemented in closeset-hit
				// if empty, closest hit will use the face normal
				vertexNormals.push_back(make_float3(0.f, 0.f, 0.f));
			}
			else {
				const glm::vec3& n = mesh.normals[v];
				vertexNormals.push_back(make_float3(n.x, n.y, n.z));
			}
		}

		if (!mesh.texcoords.empty() && mesh.texcoords.size() != mesh.positions.size()) {
			throw std::runtime_error("Texture coordinate count does not match vertex count");
		}

		for (size_t v = 0; v < mesh.positions.size(); v++) {
			if (mesh.texcoords.empty()) {
				vertexTexcoords.push_back(make_float2(0.0f, 0.0f)); //missing UVs
			}
			else {
				const glm::vec2& uv = mesh.texcoords[v];
				vertexTexcoords.push_back(make_float2(uv.x, uv.y));
			}
		}
		if (!mesh.tangents.empty() && mesh.tangents.size() != mesh.positions.size()) {
			throw std::runtime_error("Tangent count does not match vertex count");
		}

		for (size_t v = 0; v < mesh.positions.size(); v++) {
			if (mesh.tangents.empty()) {
				vertexTangents.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f));
			}
			else {
				const glm::vec4& tangent = mesh.tangents[v];
				vertexTangents.push_back(make_float4(tangent.x, tangent.y, tangent.z, tangent.w));
			}
		}

		for (const auto& t : mesh.triangles) {
			triangles.push_back(make_uint3(vertexOffset + t[0], vertexOffset + t[1], vertexOffset + t[2]));
			triangleMaterialIds.push_back(mesh.rendererMaterialId);
		}
	}

	// light above box
	const unsigned int lightVertexStart = static_cast<unsigned int>(vertices.size());
	const float lightHeight = 8.0f;
	vertices.push_back(make_float3(-2.0f, lightHeight, 1.0f));
	vertices.push_back(make_float3(2.0f, lightHeight, 1.0f));
	vertices.push_back(make_float3(2.0f, lightHeight, 5.0f));
	vertices.push_back(make_float3(-2.0f, lightHeight, 5.0f));

	// fallback for the 0s will be implemeneted in closest-hit
	for (int v = 0; v < 4; ++v) {
		vertexNormals.push_back(make_float3(0.f, 0.f, 0.f));
		vertexTexcoords.push_back(make_float2(0.0f, 0.0f));
		vertexTangents.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f));
	}

	//Two triangles forming light
	triangles.push_back(make_uint3(lightVertexStart,lightVertexStart + 1,lightVertexStart + 2));
	triangles.push_back(make_uint3(lightVertexStart,lightVertexStart + 2,lightVertexStart + 3));

	//triangles renderer material: 1
	triangleMaterialIds.push_back(lightMaterialId);
	triangleMaterialIds.push_back(lightMaterialId);

	//GPU malloc
	const size_t vertexBytes = vertices.size() * sizeof(float3);
	const size_t normalBytes = vertexNormals.size() * sizeof(float3);
	const size_t texcoordBytes = vertexTexcoords.size() * sizeof(float2);
	const size_t indexBytes = triangles.size() * sizeof(uint3);
	const size_t materialBytes = triangleMaterialIds.size() * sizeof(int);
	const size_t tangentBytes = vertexTangents.size() * sizeof(float4);

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_meshVertices), vertexBytes);
	checkCuda(cudaResult, "Mesh vertex allocation failed");

	cudaResult = cudaMemcpy(dev_meshVertices, vertices.data(), vertexBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Mesh vertex upload failed");

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_meshNormals), normalBytes);
	checkCuda(cudaResult, "Mesh normals allocation failed");

	cudaResult = cudaMemcpy(dev_meshNormals, vertexNormals.data(), normalBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Mesh normals upload failed");

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_meshTexcoords), texcoordBytes);
	checkCuda(cudaResult, "Meshtexture coordinte allocation failed");

	cudaResult = cudaMemcpy(dev_meshTexcoords, vertexTexcoords.data(), texcoordBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Mesh texture coordinate upload failed");

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_meshTangents), tangentBytes);
	checkCuda(cudaResult, "Mesh tangent allocation failed");

	cudaResult = cudaMemcpy(dev_meshTangents, vertexTangents.data(), tangentBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Mesh tangent upload failed");

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_meshIndices), indexBytes);
	checkCuda(cudaResult, "Mesh index allocation failed");

	cudaResult = cudaMemcpy(dev_meshIndices, triangles.data(), indexBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Mesh index upload failed");

	cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_triangleMaterialIds),materialBytes);
	checkCuda(cudaResult, "Triangle material allocation failed");

	cudaResult = cudaMemcpy(dev_triangleMaterialIds, triangleMaterialIds.data(), materialBytes, cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Triangle material upload failed");

	std::cout << "Uploaded " << vertices.size() << " vertices and " << triangles.size() << " triangles\n";

	//////////////////////////////////
	// Describe the triangle and query the GAS(Geometry Acceleration Structure) memory requirements
	//////////////////////////////////

	// It's like glVertexAttribPointer
	// Express GPU pointer using the address type expected by OptiX
	CUdeviceptr vertexBuffer = reinterpret_cast<CUdeviceptr>(dev_meshVertices);
	
	// One geometry-flags entry for one future hitgroup SBT record
	// No need for a any-hit program for the test triangle
	unsigned int triangleFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

	OptixBuildInput triangleInput = {}; //CPU side description of geometry used to build an acceleration structure, just a description, does not hold the vertex data
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; //tells optix that geometry consist sof triangles

	// Each vertex contains 3 floats: x,y,z
	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangleInput.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());

	// Optix expects a CPU array of GPU buffer addresses
	// With no motion blur, one address is enough
	// CPU side OptiX API is reading the address from the CPU memory, GPU doesn't repeatedly fetch that address from the CPU for every ray
	triangleInput.triangleArray.vertexBuffers = &vertexBuffer; //OptiX reads a GPU address stored in CPU memory, then uses the address to locate the vertices

	//No index buffer - every consecutive group of three vertices forms a triangle
	//Tells Optix how to read the index buffer - each index triplet selects three vertices for one triangle
	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(dev_meshIndices);
	triangleInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(triangles.size()); //12
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint3);

	// All geometry in this input uses one hitgroup SBT record - hitgroup record: a SBT record that selects the programs that handle ray intersections with taht geometry
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.flags = triangleFlags;

	//////////////////////////////////
	// Set build options and ask for the required sizes
	//////////////////////////////////

	//Build a new acceleration structure using default build flags
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	// Optix fills this with the required allocation sizes
	OptixAccelBufferSizes gasBufferSizes = {};

	optixResult = optixAccelComputeMemoryUsage(
		optixContext,
		&accelOptions,
		&triangleInput,
		1, // number of build inputs
		&gasBufferSizes
	);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX GAS memory query failed: ") +optixGetErrorString(optixResult));
	}

	// Scratch space used whiel building the GAS
	std::cout << "GAS temporary memory: " << gasBufferSizes.tempSizeInBytes << " bytes." << std::endl;
	// Storage for the completed GAS, retained while tracing
	std::cout << "GAS output memory: " << gasBufferSizes.outputSizeInBytes << " bytes." << std::endl;

	//////////////////////////////////
	// Create GAS using queried size info from above
	//////////////////////////////////

	// Allocate buffers
	// Allocate the temporary construction workspace
	cudaResult = cudaMalloc(&dev_gasTempBuffer, gasBufferSizes.tempSizeInBytes); //gasBufferSizes was populated from the stage above
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("GAS temporary allocation failed: ") +cudaGetErrorString(cudaResult));
	}

	// Allocate storage for completed GAS
	cudaResult = cudaMalloc(&dev_gasOutputBuffer, gasBufferSizes.outputSizeInBytes);
	if (cudaResult != cudaSuccess) {
		throw std::runtime_error(std::string("GAS output allocation failed: ") +cudaGetErrorString(cudaResult));
	}

	//Build the GAS
	//Build using the same optiosn and triangle description as the size query
	optixResult = optixAccelBuild(
		optixContext,
		nullptr, // use the default CUDA stream
		&accelOptions,
		&triangleInput,
		1, //one build input
		reinterpret_cast<CUdeviceptr>(dev_gasTempBuffer),
		gasBufferSizes.tempSizeInBytes,
		reinterpret_cast<CUdeviceptr>(dev_gasOutputBuffer),
		gasBufferSizes.outputSizeInBytes, 
		&gasHandle, // OptiX writes the resulting handle here
		nullptr, // No optional post-build properties requested
		0 // Number of the properties
	);

if (optixResult != OPTIX_SUCCESS) {
	throw std::runtime_error(std::string("OptiX GAS build failed: ") + optixGetErrorString(optixResult));
}

// Wait for the GPU build to finish before releasing its workspace
cudaResult = cudaDeviceSynchronize();
checkCuda(cudaResult, "GAS build synchronization failed");

// Construction is complete, so the temporary workspace is no longer needed
cudaResult = cudaFree(dev_gasTempBuffer);
if (cudaResult != cudaSuccess) {
	throw std::runtime_error(std::string("GAS temporary buffer cleanup failed: ") + cudaGetErrorString(cudaResult));
}
dev_gasTempBuffer = nullptr;

std::cout << "Optix mesh GAS built." << std::endl;

//////////////////////////////////
// Uplaod parameters after building the GAS
//////////////////////////////////
//Prepare the parameter values on the CPU
#if TEST_LAUNCH
LaunchParams launchParams = {};
launchParams.gasHandle = gasHandle;
launchParams.vertices = dev_meshVertices;
launchParams.triangles = dev_meshIndices;
#endif

//Allocate a GPU buffer for those values
cudaResult = cudaMalloc(reinterpret_cast<void**>(&dev_launchParams), sizeof(LaunchParams));
checkCuda(cudaResult, "Launch parameter allocation failed");

//Compare it with the value printed by raygen
std::cout << "CPU GAS handle: " << static_cast<unsigned long long>(gasHandle) << std::endl;

//Raygen Implementation
initOptixPathtracePipeline();
}

//Raygen Implementation, same as launchOptixIntersections
void launchOptixPaths(
	PathSegment* paths, ShadeableIntersection* intersections,
	const Material* materials,
	int numPaths, int iteration,
	const Camera& camera, int maxBounces, bool generateCameraRays) {
	if (numPaths <= 0) {
		return;
	}

	if (pathtracePipeline == nullptr || dev_launchParams == nullptr) {
		throw std::runtime_error("OptiX pathtrace pipeline has not been initialized");
	}

	if (paths == nullptr || intersections == nullptr || materials == nullptr) {
		throw std::runtime_error("OptiX pathtrace received a null buffer");
	}

	LaunchParams launchParams = {};

	launchParams.gasHandle = gasHandle;
	launchParams.vertices = dev_meshVertices;
	launchParams.normals = dev_meshNormals;
	launchParams.tangents = dev_meshTangents;
	
	launchParams.texcoords = dev_meshTexcoords;
	launchParams.textures = dev_textureObjects;
	
	launchParams.environmentTexture = environmentTexture;
	launchParams.environmentCdf = dev_environmentCdf;
	launchParams.environmentWidth = environmentWidth;
	launchParams.environmentHeight = environmentHeight;

	launchParams.triangles = dev_meshIndices;
	launchParams.triangleMaterialIds = dev_triangleMaterialIds;

	launchParams.paths = paths;
	launchParams.outputPaths = paths;
	launchParams.intersections = intersections;
	launchParams.materials = materials;
	launchParams.numPaths = static_cast<unsigned int>(numPaths);
	launchParams.iteration = iteration;

	launchParams.camera.width = camera.resolution.x;
	launchParams.camera.height = camera.resolution.y;
	launchParams.camera.position = make_float3(camera.position.x, camera.position.y, camera.position.z);
	launchParams.camera.view = make_float3(camera.view.x, camera.view.y, camera.view.z);
	launchParams.camera.up = make_float3(camera.up.x, camera.up.y, camera.up.z);
	launchParams.camera.right = make_float3(camera.right.x, camera.right.y, camera.right.z);
	launchParams.camera.pixelLength = make_float2(camera.pixelLength.x, camera.pixelLength.y);
	launchParams.camera.apertureRadius = camera.apertureRadius;
	launchParams.camera.focalDistance = camera.focalDistance;
	
	launchParams.maxBounces = maxBounces;
	launchParams.generateCameraRays = generateCameraRays ? 1 : 0;

	cudaError_t cudaResult = cudaMemcpy(dev_launchParams, &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Upload pathtrace launch parameters");

	OptixResult optixResult = optixLaunch(
		pathtracePipeline,
		nullptr,
		reinterpret_cast<CUdeviceptr>(dev_launchParams),
		sizeof(LaunchParams),
		&pathtraceSbt,
		launchParams.numPaths,
		1,
		1);
	checkOptix(optixResult, "Launch OptiX pathtrace");
}

// Preparing bridge between CUDA renderer and OptiX - passes the GPu buffer addresses, GAS handle, and path count to OptiX
void launchOptixIntersections(const PathSegment* paths, ShadeableIntersection* intersections, int numPaths) {
	// Nothing to launch once all paths have terminated
	if (numPaths <= 0){
		return;
	}

	// Must have been created during initialization
	if (optixPipeline == nullptr || dev_launchParams == nullptr) {
		throw std::runtime_error("OptiX has not been initialized");
	}
	if (paths == nullptr || intersections == nullptr) {
		throw std::runtime_error("OptiX received a null ray or intersection buffer");
	}

	//Only traces existing rays and writes intersections, so it doesn't need materials, camera settings, or output paths unlike luanchOptixPaths
	//Prepare the launch's addresses and active-path count on the CPU
	//Here are my GPu rays, here is where their intersection results should go, and here is how many rays to process
	LaunchParams launchParams = {};
	launchParams.gasHandle = gasHandle;
	launchParams.vertices = dev_meshVertices;
	launchParams.normals = dev_meshNormals;
	launchParams.tangents = dev_meshTangents;
	launchParams.texcoords = dev_meshTexcoords; //UV buffer
	launchParams.textures = dev_textureObjects;
	launchParams.triangles = dev_meshIndices;
	launchParams.triangleMaterialIds = dev_triangleMaterialIds;

	launchParams.paths = paths;
	launchParams.intersections = intersections;
	launchParams.numPaths = static_cast<unsigned int>(numPaths);

	//Reuse the parameter allocation created by initOptixContext()
	cudaError_t cudaResult = cudaMemcpy(dev_launchParams, &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice);
	checkCuda(cudaResult, "Launch parameter upload failed");

	//////////////////////////////////
	// Launch pipeline
	//////////////////////////////////
	// One raygen invocation per active path
	OptixResult optixResult = optixLaunch(
		optixPipeline,
		nullptr,
		reinterpret_cast<CUdeviceptr>(dev_launchParams),
		sizeof(LaunchParams),
		&sbt,
		launchParams.numPaths,
		1,
		1
	);
	checkOptix(optixResult, "OptiX intersection launch failed");

	// Catch GPU execution errors during initial integration
	//cudaResult = cudaDeviceSynchronize();
	checkCuda(cudaResult, "OptiX intersection execution failed");
}

void destroyOptixContext() {
	// Raygen Implementation
	// Complete remnant launches before destroying resources
	cudaError_t syncResult = cudaDeviceSynchronize();
	checkCuda(syncResult, "Synchronize before OptiX cleanup");

	if (dev_environmentCdf != nullptr) {
		cudaError_t result = cudaFree(dev_environmentCdf);
		checkCuda(result, "Environment CDF cleanup failed");
		dev_environmentCdf = nullptr;
	}
	environmentWidth = 0;
	environmentHeight = 0;

	if (environmentTexture != 0) {
		cudaError_t result = cudaDestroyTextureObject(environmentTexture);
		checkCuda(result, "Environment texture cleanup failed");
		environmentTexture = 0;
	}

	if (dev_environmentArray != nullptr) {
		cudaError_t result = cudaFreeArray(dev_environmentArray);
		checkCuda(result, "Environment aray cleanup failed");
		dev_environmentArray = nullptr;
	}

	if (dev_textureObjects != nullptr) {
		cudaError_t result = cudaFree(dev_textureObjects);
		checkCuda(result, "Texture handle cleanup failed");
		dev_textureObjects = nullptr;
	}

	for (cudaTextureObject_t& textureObject : textureObjects) {
		if (textureObject != 0) {
			cudaError_t result = cudaDestroyTextureObject(textureObject);
			checkCuda(result, "Texture object cleanup failed");
			textureObject = 0;
		}
	}
	textureObjects.clear();

	for (cudaArray_t& imageArray : dev_imageArrays) {
		if (imageArray != nullptr) {
			cudaError_t result = cudaFreeArray(imageArray);
			checkCuda(result, "Image array cleanup failed");
			imageArray = nullptr;
		}
	}
	dev_imageArrays.clear();

	if (dev_meshTangents != nullptr) {
		cudaError_t result = cudaFree(dev_meshTangents);
		checkCuda(result, "Mesh tangent cleanup failed");
		dev_meshTangents = nullptr;
	}

	if (pathtracePipeline != nullptr) {
		OptixResult result = optixPipelineDestroy(pathtracePipeline);
		checkOptix(result, "Destroy pathtrace pipeline");
		pathtracePipeline = nullptr;
	}

	if (dev_pathtraceRaygenRecord != nullptr) {
		cudaError_t result = cudaFree(dev_pathtraceRaygenRecord);
		checkCuda(result, "Free pathtrace raygen record");
		dev_pathtraceRaygenRecord = nullptr;
	}

	pathtraceSbt = {};

	if (pathtraceRaygenProgramGroup != nullptr) {
		OptixResult result =
			optixProgramGroupDestroy(pathtraceRaygenProgramGroup);

		checkOptix(result, "Destroy pathtrace raygen program group");
		pathtraceRaygenProgramGroup = nullptr;
	}

	// The test launch as already snychronized before cleanup.
	if (dev_launchParams != nullptr) {
		cudaError_t result = cudaFree(dev_launchParams);
		checkCuda(result, "Launch parameter cleanup failed");
		dev_launchParams = nullptr;
	}

	// Release gas tempbuffer
	if (dev_gasTempBuffer != nullptr) {
		cudaError_t result = cudaFree(dev_gasTempBuffer);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("GAS temporary buffer cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_gasTempBuffer = nullptr;
	}

	// All GPU work using the GAS must finish before the allocation is freed, our current test already synchronizes before cleanup
	if (dev_gasOutputBuffer != nullptr) {
		cudaError_t result = cudaFree(dev_gasOutputBuffer);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("GAS output buffer cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_gasOutputBuffer = nullptr;
		gasHandle = 0;
	}

	// Release mesh triangle GPU vertex allocation
	if (dev_meshVertices != nullptr) {
		cudaError_t result = cudaFree(dev_meshVertices);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("Triangle vertex cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_meshVertices = nullptr;
	}

	if (dev_meshNormals != nullptr) {
		cudaError_t result = cudaFree(dev_meshNormals);
		checkCuda(result, "Mesh normal cleanup failed");
		dev_meshNormals = nullptr;
	}

	if (dev_meshTexcoords != nullptr) {
		cudaError_t result = cudaFree(dev_meshTexcoords);
		checkCuda(result, "Mesh texture cordinate cleanup failed");
		dev_meshTexcoords = nullptr;
	}

	// Release mesh triaingle GPU indices allocation
	if (dev_meshIndices != nullptr) {
		cudaError_t result = cudaFree(dev_meshIndices);
		checkCuda(result, "Mesh index cleanup failed");
		dev_meshIndices = nullptr;
	}

	// Release triangle material ids
	if (dev_triangleMaterialIds != nullptr) {
		cudaError_t result = cudaFree(dev_triangleMaterialIds);
		checkCuda(result, "Triangle material cleanup failed");
		dev_triangleMaterialIds = nullptr;
	}

	// Release hit record
	if (dev_hitgroupRecord != nullptr){
		cudaError_t result = cudaFree(dev_hitgroupRecord);
		checkCuda(result, "Hitgroup record cleanup failed");
		dev_hitgroupRecord = nullptr;
		sbt.hitgroupRecordBase = 0;
		sbt.hitgroupRecordStrideInBytes = 0;
		sbt.hitgroupRecordCount = 0;
	}

	// Release miss record
	if (dev_missRecord != nullptr) {
		cudaError_t result = cudaFree(dev_missRecord);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("Miss record cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_missRecord = nullptr;
		sbt.missRecordBase = 0;
		sbt.missRecordStrideInBytes = 0;
		sbt.missRecordCount = 0;
	}

	// Release the GPU memory allcoated for the raygen record
	if (dev_raygenRecord != nullptr) {
		cudaError_t result = cudaFree(dev_raygenRecord);
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string("Raygen record cleanup failed: ") + cudaGetErrorString(result));
		}
		dev_raygenRecord = nullptr;
		sbt.raygenRecord = 0;
	}

	// Release the pipeline before cleaning up the other OptiX object
	if (optixPipeline != nullptr) {
		OptixResult result = optixPipelineDestroy(optixPipeline);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("OptiX pipeline destruction failed: ") + optixGetErrorString(result));
		}
		optixPipeline = nullptr;
	}

	// Release hit program group
	if (hitgroupProgramGroup != nullptr) {
		OptixResult result = optixProgramGroupDestroy(hitgroupProgramGroup);
		checkOptix(result, "Hitgroup program group destruction failed");
		hitgroupProgramGroup = nullptr;
	}

	//Release miss program group
	if (missProgramGroup != nullptr) {
		OptixResult result = optixProgramGroupDestroy(missProgramGroup);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("Miss program group destruction failed: ") + optixGetErrorString(result));
		}
		missProgramGroup = nullptr;
	}
	
	//Release the program group before releasing its module
	if (raygenProgramGroup != nullptr) {
		OptixResult result = optixProgramGroupDestroy(raygenProgramGroup);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("OptiX program group destruction failed: ") + optixGetErrorString(result));
		}
		raygenProgramGroup = nullptr;
	}

	//Release optix module
	if (optixModule != nullptr) {
		OptixResult result = optixModuleDestroy(optixModule);
		if (result != OPTIX_SUCCESS) {
			throw std::runtime_error(std::string("OptiX module destruction failed: ") + optixGetErrorString(result));
		}
		optixModule = nullptr;
	}

	// Nothing to destroy if no context data
	if (optixContext == nullptr) {
		return;
	}
	// Destroy the OptiX context using its handle
	OptixResult optixResult = optixDeviceContextDestroy(optixContext);
	if (optixResult != OPTIX_SUCCESS) {
		throw std::runtime_error(std::string("OptiX context destruction failed: ") + optixGetErrorString(optixResult));
	}
	// Old handle is no longer valid, clear to make repeated cleanup calls harmless
	optixContext = nullptr;
}