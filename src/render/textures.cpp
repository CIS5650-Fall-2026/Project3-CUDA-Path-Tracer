#include "render/textures.h"

#include "scene/scene.h"
#include "utilities.h"

#include <cmath>
#include <vector>

namespace
{
std::vector<cudaArray_t> arrays;                     // one per Scene::textureImages entry
std::vector<cudaTextureObject_t> objects;            // one per Scene::textures entry
cudaTextureObject_t* dev_textureObjects = nullptr;
cudaArray_t environmentArray = nullptr;
cudaTextureObject_t environmentObject = 0;
const float* environmentTables[3] = {};  // pdfUv, conditionalCdf, marginalCdf
}  // namespace

cudaTextureObject_t* texturesInit(const Scene& scene)
{
    if (scene.textures.empty())
    {
        return nullptr;
    }

    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
    for (const TextureImage& image : scene.textureImages)
    {
        cudaArray_t array = nullptr;
        CUDA_CHECK(cudaMallocArray(&array, &format, image.width, image.height));
        const size_t rowBytes = (size_t)image.width * 4;
        CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, image.rgba.data(), rowBytes, rowBytes, image.height,
            cudaMemcpyHostToDevice));
        arrays.push_back(array);
    }

    for (const Texture& texture : scene.textures)
    {
        cudaResourceDesc resource = {};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = arrays[texture.image];

        cudaTextureDesc desc = {};
        desc.addressMode[0] = texture.wrapU;
        desc.addressMode[1] = texture.wrapV;
        desc.filterMode = texture.filter;
        desc.readMode = cudaReadModeNormalizedFloat;
        desc.sRGB = texture.srgb ? 1 : 0;
        desc.normalizedCoords = 1;

        cudaTextureObject_t object = 0;
        CUDA_CHECK(cudaCreateTextureObject(&object, &resource, &desc, nullptr));
        objects.push_back(object);
    }

    CUDA_CHECK(cudaMalloc(&dev_textureObjects, objects.size() * sizeof(cudaTextureObject_t)));
    CUDA_CHECK(cudaMemcpy(dev_textureObjects, objects.data(), objects.size() * sizeof(cudaTextureObject_t),
        cudaMemcpyHostToDevice));
    return dev_textureObjects;
}

void texturesFree()
{
    for (cudaTextureObject_t object : objects)
    {
        cudaDestroyTextureObject(object);
    }
    for (cudaArray_t array : arrays)
    {
        cudaFreeArray(array);
    }
    cudaFree(dev_textureObjects);
    objects.clear();
    arrays.clear();
    dev_textureObjects = nullptr;
}

EnvironmentMap environmentInit(const Environment& env)
{
    EnvironmentMap map = {};
    map.radiance = env.radiance;
    map.cosRotation = std::cos(env.rotation);
    map.sinRotation = std::sin(env.rotation);
    if (env.rgba.empty())
    {
        return map;
    }

    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&environmentArray, &format, env.width, env.height));
    const size_t rowBytes = (size_t)env.width * sizeof(float4);
    CUDA_CHECK(cudaMemcpy2DToArray(environmentArray, 0, 0, env.rgba.data(), rowBytes, rowBytes, env.height,
        cudaMemcpyHostToDevice));

    cudaResourceDesc resource = {};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = environmentArray;

    cudaTextureDesc desc = {};
    desc.addressMode[0] = cudaAddressModeWrap;   // u runs around the horizon
    desc.addressMode[1] = cudaAddressModeClamp;  // v stops at the poles
    desc.filterMode = cudaFilterModeLinear;
    desc.readMode = cudaReadModeElementType;
    desc.normalizedCoords = 1;
    CUDA_CHECK(cudaCreateTextureObject(&environmentObject, &resource, &desc, nullptr));
    map.texture = environmentObject;

    // An all-black map has no sampling table
    if (!env.pdfUv.empty())
    {
        const auto upload = [](const std::vector<float>& v) {
            float* d = nullptr;
            CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
            return d;
        };
        map.pdfUv = upload(env.pdfUv);
        map.conditionalCdf = upload(env.conditionalCdf);
        map.marginalCdf = upload(env.marginalCdf);
        map.width = env.width;
        map.height = env.height;
        environmentTables[0] = map.pdfUv;
        environmentTables[1] = map.conditionalCdf;
        environmentTables[2] = map.marginalCdf;
    }
    return map;
}

void environmentFree()
{
    if (environmentObject != 0)
    {
        cudaDestroyTextureObject(environmentObject);
    }
    cudaFreeArray(environmentArray);
    environmentObject = 0;
    environmentArray = nullptr;
    for (const float*& table : environmentTables)
    {
        cudaFree((void*)table);
        table = nullptr;
    }
}
