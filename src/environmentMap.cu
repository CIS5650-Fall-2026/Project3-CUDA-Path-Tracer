#include "environmentMap.h"

#include "cudaHelpers.h"
#include "textureLoader.h"

#include <glm/gtc/constants.hpp>

__device__ glm::vec3 DevEnvironmentMap::sample(const glm::vec3& direction) const {
    if (texture == 0) {
        return glm::vec3(0.0f);
    }

    float theta = glm::acos(direction.y);
    float phi = atan2f(direction.z, direction.x);

    float u = 1.0f - (phi + glm::pi<float>()) / (2.0f * glm::pi<float>());
    float v = theta / glm::pi<float>();

    float4 sampled = tex2D<float4>(texture, u, v);
    return intensity * glm::vec3(sampled.x, sampled.y, sampled.z);
}

EnvironmentMap::EnvironmentMap(float* imageData, size_t width, size_t height)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&m_environmentMapArray, &channelDesc, width, height));

    size_t pitch = width * sizeof(float) * 4;
    CUDA_CHECK(cudaMemcpy2DToArray(
        m_environmentMapArray,
        0, 0,
        imageData,
        pitch,
        pitch,
        height,
        cudaMemcpyHostToDevice
    ));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_environmentMapArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Wrap U for spherical map
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp V for poles
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    if (cudaCreateTextureObject(&m_environmentMapTexture, &resDesc, &texDesc, nullptr) != cudaSuccess) {
        cudaFreeArray(m_environmentMapArray);
        m_environmentMapArray = nullptr;
        m_environmentMapTexture = 0;
    }
}

EnvironmentMap::EnvironmentMap(const std::string& filepath)
{
    int width, height, channels;
    // Force 4 channels for cuda
    float* imageData = stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);

    if (!imageData) {
		std::cerr << "Failed to load environment map: " << filepath << std::endl;
        return;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&m_environmentMapArray, &channelDesc, width, height));

    size_t pitch = width * sizeof(float) * 4;
    CUDA_CHECK(cudaMemcpy2DToArray(
        m_environmentMapArray,
        0, 0,
        imageData,
        pitch,
        pitch,
        height,
        cudaMemcpyHostToDevice
    ));

	stbi_image_free(imageData);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_environmentMapArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Wrap U for spherical map
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp V for poles
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    if (cudaCreateTextureObject(&m_environmentMapTexture, &resDesc, &texDesc, nullptr) != cudaSuccess) {
        cudaFreeArray(m_environmentMapArray);
        m_environmentMapArray = nullptr;
        m_environmentMapTexture = 0;
    }
}

EnvironmentMap::~EnvironmentMap()
{
    if (m_environmentMapTexture != 0) {
        cudaDestroyTextureObject(m_environmentMapTexture);
        m_environmentMapTexture = 0;
    }

    if (m_environmentMapArray != nullptr) {
        cudaFreeArray(m_environmentMapArray);
        m_environmentMapArray = nullptr;
    }
}

EnvironmentMap& EnvironmentMap::operator=(EnvironmentMap&& other) noexcept
{
    if (this != &other) {
        if (m_environmentMapTexture != 0) {
            cudaDestroyTextureObject(m_environmentMapTexture);
            m_environmentMapTexture = 0;
        }

        if (m_environmentMapArray != nullptr) {
            cudaFreeArray(m_environmentMapArray);
            m_environmentMapArray = nullptr;
        }

        m_environmentMapTexture = other.m_environmentMapTexture;
        m_environmentMapArray = other.m_environmentMapArray;
        m_intensity = other.m_intensity;

        other.m_environmentMapTexture = 0;
        other.m_environmentMapArray = nullptr;
    }

    return *this;
}

EnvironmentMap::EnvironmentMap(EnvironmentMap&& other) noexcept :
    m_environmentMapTexture(other.m_environmentMapTexture),
    m_intensity(other.m_intensity),
    m_environmentMapArray(other.m_environmentMapArray)
{
    other.m_environmentMapTexture = 0;
    other.m_environmentMapArray = nullptr;
}
