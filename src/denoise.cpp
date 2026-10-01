#include "denoise.h"

#include <iostream>

#ifdef USE_OIDN
#include <OpenImageDenoise/oidn.hpp>

static oidn::DeviceRef makeDevice()
{
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    device.commit();
    return device;
}
#endif

void denoise(RenderState& state, int iteration)
{
    int width = state.camera.resolution.x;
    int height = state.camera.resolution.y;
    std::vector<glm::vec3> color(state.image.size());
    std::vector<glm::vec3> albedo(state.image.size());
    std::vector<glm::vec3> normal(state.image.size());
    for (int i = 0; i < state.image.size(); i++)
    {
        color[i] = state.image[i] / (float)iteration;
        albedo[i] = state.albedo[i] / (float)iteration;
        normal[i] = state.normal[i] / (float)iteration;
    }

#ifdef USE_OIDN
    static oidn::DeviceRef device = makeDevice();

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", color.data(), oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedo.data(), oidn::Format::Float3, width, height);
    filter.setImage("normal", normal.data(), oidn::Format::Float3, width, height);
    filter.setImage("output", state.denoised.data(), oidn::Format::Float3, width, height);
    filter.set("hdr", true);
    filter.commit();
    filter.execute();

    const char* error;
    if (device.getError(error) != oidn::Error::None)
    {
        std::cout << "OIDN error: " << error << std::endl;
    }
#else
    state.denoised = color;
#endif
}
