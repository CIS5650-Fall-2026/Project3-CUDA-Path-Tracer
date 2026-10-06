#include "denoiser.h"

static oidn::DeviceRef device;
static oidn::FilterRef filter;

void initDenoiser(int width, int height, glm::vec3* d_image, glm::vec3* d_denoised)
{
	// default 
	int cudaDeviceId = 0;
	cudaStream_t stream = 0;

	device = oidn::newCUDADevice(cudaDeviceId, stream);
	device.commit();

	// input and output
	oidn::BufferRef img = device.newBuffer(d_image, width * height * sizeof(glm::vec3));
	oidn::BufferRef denoisedImg = device.newBuffer(d_denoised, width * height * sizeof(glm::vec3));

	filter = device.newFilter("RT");
	filter.setImage("color", img, oidn::Format::Float3, width, height);
	filter.setImage("output", denoisedImg, oidn::Format::Float3, width, height);
	filter.set("hdr", true);
	filter.commit();
}

void denoise() {
	filter.execute();
	device.sync();
}

void denoiserFree() {
	device = nullptr;
	filter = nullptr;
}