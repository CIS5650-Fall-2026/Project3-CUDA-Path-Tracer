#include "vdb_loading.h"

#include <glm/glm.hpp>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/io/IO.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/HandleStorage.h>

#include <string>

using namespace glm;

void CreateTestNVDB(mat4* invTransform, void** d_grid) {
	nanovdb::GridHandle<nanovdb::HostBuffer> hostHandle = 
		nanovdb::tools::createFogVolumeSphere<float>(0.5f, nanovdb::Vec3f(0.0f, 0.0f, 0.0f), 0.01f, 3.0f);
	
	*d_grid = NULL;
	cudaMalloc(d_grid, hostHandle.size());
	cudaMemcpy(*d_grid, hostHandle.data(), hostHandle.size(), cudaMemcpyHostToDevice);
	 
	// BBX
	auto grid = hostHandle.grid<float>();
	auto rootBbx = grid->worldBBox();
	vec3 bbxMin = vec3(rootBbx.min()[0], rootBbx.min()[1], rootBbx.min()[2]);
	vec3 bbxMax = vec3(rootBbx.max()[0], rootBbx.max()[1], rootBbx.max()[2]);

	vec3 avg = 0.5f * (bbxMin + bbxMax);
	vec3 scale = bbxMax - bbxMin;

	*invTransform = glm::inverse(glm::translate(mat4(1.0f), avg) * glm::scale(mat4(1.0f), scale));
}

bool LoadNVDB(const std::string& fileName, mat4* invTransform, void** d_grid, std::string gridName) {
	//auto gridHandle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(fileName);
	//if (!gridHandle.grid<float>()) throw std::runtime_error(fileName + " doesn't exist or contain float grid!");
	//gridHandle.deviceUpload(); // Upload to device
	//*d_grid = gridHandle.deviceGrid<float>();

	if (!nanovdb::io::hasGrid(fileName, gridName))
		return false;

	auto hostHandle = nanovdb::io::readGrid<nanovdb::HostBuffer>(fileName, gridName);

	*d_grid = NULL;
	cudaMalloc(d_grid, hostHandle.size());
	cudaMemcpy(*d_grid, hostHandle.data(), hostHandle.size(), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	    
	// BBX
	auto grid = hostHandle.grid<float>();
	auto rootBbx = grid->worldBBox();
	vec3 bbxMin = vec3(rootBbx.min()[0], rootBbx.min()[1], rootBbx.min()[2]);
	vec3 bbxMax = vec3(rootBbx.max()[0], rootBbx.max()[1], rootBbx.max()[2]);
	
	vec3 avg = 0.5f * (bbxMin + bbxMax);
	vec3 scale = bbxMax - bbxMin;
	
	*invTransform = glm::inverse(glm::translate(mat4(1.0f), avg) * glm::scale(mat4(1.0f), scale));

	return true;
}