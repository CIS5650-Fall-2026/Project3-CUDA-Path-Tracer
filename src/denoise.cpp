#include "denoise.h"

#include <cstdio>

#ifdef USE_OIDN

#include <cuda_runtime.h>
#include <OpenImageDenoise/oidn.hpp>

// High looks best, but Balanced is cheaper per frame, so do Balanced for realtime, High for render
#define DENOISE_QUALITY oidn::Quality::High

namespace
{
    oidn::DeviceRef device;
    oidn::FilterRef filter;
    bool ready = false;

    // Prints and returns true if device has error
    bool deviceFailed(const char* what)
    {
        const char* message = nullptr;
        if (device.getError(message) != oidn::Error::None)
        {
            fprintf(stderr, "OIDN error (%s): %s\n", what, message ? message : "unknown");
            return true;
        }
        return false;
    }
}

bool denoiserAvailable()
{
    return ready;
}

bool denoiserInit(int width, int height, float* color, float* albedo, float* normal, float* output)
{
    denoiserFree();

    int cudaDevice = 0;
    cudaGetDevice(&cudaDevice);
    if (!oidn::isCUDADeviceSupported(cudaDevice))
    {
        fprintf(stderr, "OIDN: CUDA device %d is not supported, denoiser disabled\n", cudaDevice);
        return false;
    }

    // By using the same stream as the CUDA default stream (nullptr), it will force
    // this command to happen after the last CUDA call without synchronization.
    device = oidn::newCUDADevice(cudaDevice, nullptr);
    device.commit();
    if (deviceFailed("device"))
    {
        return false;
    }

    // Buffers are read and written in place, so we don't need to copy anything
    filter = device.newFilter("RT");
    filter.setImage("color",  color,  oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedo, oidn::Format::Float3, width, height);
    filter.setImage("normal", normal, oidn::Format::Float3, width, height);
    filter.setImage("output", output, oidn::Format::Float3, width, height);
    // assumign we can have high definition range colors outside of [0, 1] range
    filter.set("hdr", true);
    // Our albedo and normal textures are already noise free, so we can
    // pass this flag to avoid denoising them again before using them
    filter.set("cleanAux", true);
    filter.set("quality", DENOISE_QUALITY);
    filter.commit();
    if (deviceFailed("filter"))
    {
        return false;
    }

    ready = true;
    return true;
}

void denoiserRun()
{
    if (!ready)
    {
        return;
    }
    filter.execute();
    deviceFailed("execute");
}

void denoiserFree()
{
    filter.release();
    device.release();
    ready = false;
}

#else // !USE_OIDN

bool denoiserAvailable() { return false; }
bool denoiserInit(int, int, float*, float*, float*, float*) { return false; }
void denoiserRun() {}
void denoiserFree() {}

#endif
