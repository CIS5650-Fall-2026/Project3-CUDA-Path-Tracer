#include <optix.h>
#include <stdio.h>
#include <cuda_runtime.h> //helps intellisense
#include <optix_device.h> //GPU side optix functions, including optixTrace
#include "optixLaunchParams.h"
#include "sceneStructs.h" // Members of Pathsegment and ShadeableIntersection
#include <float.h>
#include <glm/glm.hpp>
#include <thrust/random.h>
#include "intersections.h"
#include "utilities.h"

#define USE_ENVIRONMENT_MIS 1

// OptiX supplies the variable's contents when it launches
// extern "C" preserves the exact symbol name - params
extern "C" {
	__constant__ LaunchParams params;
}

// extern "C" preserves the function name for lookup by OptiX - no mangling
// __raygen__ is the required prefix for an OptiX ray-generation program
extern "C" __global__ void __raygen__rg() {
	// Reads paths[index].ray and calls optixTrace
	// The launch is one-dimensional: each x index handles one active path
	const unsigned int index = optixGetLaunchIndex().x; //like thread index?

	if (index >= params.numPaths) {
		return;
	}

	// ADD
	if (params.paths[index].remainingBounces <= 0) {
		return;
	}

	const Ray& ray = params.paths[index].ray;

	const float3 origin = make_float3(ray.origin.x, ray.origin.y, ray.origin.z);
	const float3 direction = make_float3(ray.direction.x, ray.direction.y, ray.direction.z);

	//Traces 1 ray against the geometry in the GAS
	optixTrace(
		params.gasHandle,
		origin,
		direction,
		0.001f, // Ignore intersections extremely near the origin
		FLT_MAX, //Maximum ray parameter
		0.0f, //No motion blur
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0, //Hitgroup SBT offset
		1, //Hitgroup SBT stride
		0 //Miss record index
	);
}

// Runs when traversal finds the closest accepted intersection
extern "C" __global__ void __closesthit__ch() {
	const unsigned int index = optixGetLaunchIndex().x; //index'th thread(ray)
	ShadeableIntersection& result = params.intersections[index]; //index'th intersection

	result.t = optixGetRayTmax(); //noice

	const unsigned int primitiveIndex = optixGetPrimitiveIndex(); //got index of triangle that was hit
	const uint3 triangle = params.triangles[primitiveIndex];  //the triangle the ray hit, has 3 vertex indices, not 3 positions

    const float2 bary = optixGetTriangleBarycentrics();
    
    const float w0 = 1.0f - bary.x - bary.y;
    const float w1 = bary.x;
    const float w2 = bary.y;

    result.texcoord = glm::vec2(0.0f);

    if (params.texcoords != nullptr) {
        const float2 uv0 = params.texcoords[triangle.x];
        const float2 uv1 = params.texcoords[triangle.y];
        const float2 uv2 = params.texcoords[triangle.z];

        result.texcoord = w0 * glm::vec2(uv0.x, uv0.y) + w1 * glm::vec2(uv1.x, uv1.y) + w2 * glm::vec2(uv2.x, uv2.y);
    }

	const float3 a = params.vertices[triangle.x]; //triangle holds indices to each vertex, x y z are each indices of a vertex 
	const float3 b = params.vertices[triangle.y];
	const float3 c = params.vertices[triangle.z];

	const glm::vec3 p0(a.x, a.y, a.z);
	const glm::vec3 p1(b.x, b.y, b.z);
	const glm::vec3 p2(c.x, c.y, c.z);

	//Geometric normal perpendicular to the triangle
	glm::vec3 geometricNormal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    glm::vec3 normal = geometricNormal;

	const float3 na = params.normals[triangle.x];
	const float3 nb = params.normals[triangle.y];
	const float3 nc = params.normals[triangle.z];

	const glm::vec3 n0(na.x, na.y, na.z);
	const glm::vec3 n1(nb.x, nb.y, nb.z);
	const glm::vec3 n2(nc.x, nc.y, nc.z);

	//Use vertex noramls only when all three are present
	if (glm::dot(n0, n0) > 0.0f && glm::dot(n1, n1) > 0.0f && glm::dot(n2, n2) > 0.0f) {
		const glm::vec3 interpolatedNormal = w0 * n0 + w1 * n1 + w2 * n2;

		if (glm::dot(interpolatedNormal, interpolatedNormal) > 0.0f) {
			normal = glm::normalize(interpolatedNormal);
		}
	}

    result.surfaceTangent = glm::vec3(0.0f);
    result.surfaceBitangent = glm::vec3(0.0f);

    glm::vec3 tangent(0.0f);
    float handedness = 1.0f;
    bool hasTangent = false;

    if (params.tangents != nullptr) {
        const float4 t0 = params.tangents[triangle.x];
        const float4 t1 = params.tangents[triangle.y];
        const float4 t2 = params.tangents[triangle.z];

        //W = 0 - marker for missing tangents
        if (t0.w != 0.0f && t1.w != 0.0f && t2.w != 0.0f) {
            tangent = w0 * glm::vec3(t0.x, t0.y, t0.z) +
                w1 * glm::vec3(t1.x, t1.y, t1.z) +
                w2 * glm::vec3(t2.x, t2.y, t2.z);

            float interpolatedSign = w0 * t0.w + w1 * t1.w + w2 * t2.w;

            handedness = interpolatedSign < 0.0f ? -1.0f : 1.0f; //checks for mirrored bitangent (if weighted sum of bary weight is negative) in case uv is flipped or mirrored
            hasTangent = true;
        }
    }

    if (!hasTangent && params.texcoords != nullptr) {
        //derives directions from how the triangle's world space edges correspond to its uv edges
        const float2 uv0 = params.texcoords[triangle.x];
        const float2 uv1 = params.texcoords[triangle.y];
        const float2 uv2 = params.texcoords[triangle.z];

        const glm::vec2 deltaUV1(uv1.x - uv0.x, uv1.y - uv0.y);
        const glm::vec2 deltaUV2(uv2.x - uv0.x, uv2.y - uv0.y);

        const glm::vec3 edge1 = p1 - p0;
        const glm::vec3 edge2 = p2 - p0;

        const float determinant = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;

        if (fabsf(determinant) > 0.000000000001f) {
            tangent = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) / determinant;
            const glm::vec3 bitangent = (edge2 * deltaUV1.x - edge1 * deltaUV2.x) / determinant;

            handedness = glm::dot(glm::cross(normal, tangent), bitangent) < 0.0f ? -1.0f : 1.0f;

            hasTangent = true;
        }
    }

    //inerpolation can make the tangent slightly nonperpendicular to the normal
    if (hasTangent) {
        //Remove the component pointing along the normal
        tangent -= normal * glm::dot(normal, tangent);

        if (glm::dot(tangent, tangent) > 0.000000000001f) {
            result.surfaceTangent = glm::normalize(tangent);
            result.surfaceBitangent = handedness * glm::cross(normal, result.surfaceTangent);
        }
    }

	//Forced flat shading
	//normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
	//Orient normal against incoming ray
	const float3 direction = optixGetWorldRayDirection();
	const glm::vec3 rayDirection(direction.x, direction.y, direction.z);

    result.geometricNormal = geometricNormal;

	if (glm::dot(normal, rayDirection) > 0.000000000001f) {
		normal = -normal;

        result.surfaceTangent = -result.surfaceTangent;
        result.surfaceBitangent = -result.surfaceBitangent;
	}

	result.surfaceNormal = normal;

	//Triangle render materials
	result.materialId = params.triangleMaterialIds[primitiveIndex];
}

// Runs when traversal finds no intersection
extern "C" __global__ void __miss__ms() {
	const unsigned int index = optixGetLaunchIndex().x;
	ShadeableIntersection& result = params.intersections[index];

	result.t = -1.0f;
    result.geometricNormal = glm::vec3(0.0f);

	// No material or surface exists
	result.materialId = -1;
	result.surfaceNormal.x = 0.0f;
	result.surfaceNormal.y = 0.0f;
	result.surfaceNormal.z = 0.0f;

    result.texcoord = glm::vec2(0.0f);
    result.surfaceTangent = glm::vec3(0.0f);
    result.surfaceBitangent = glm::vec3(0.0f);
	
	//printf("Ray missed\n");
}


//Raygen Loop Implementation
static __forceinline__ __device__
thrust::default_random_engine raygenMakeRandomEngine(int iteration, int pixelIndex, int remainingBounces) {
    // Same as the CPU-loop
    int h = utilhash((1 << 31) | (remainingBounces << 22) | iteration) ^ utilhash(pixelIndex);
    return thrust::default_random_engine(h);
}

static __forceinline__ __device__
glm::vec3 raygenSampleHemisphere( glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));

    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

static __forceinline__ __device__
float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    //positive incident cosine and the correct incident/transmitted indices of refraction
    //cosThetaI = cos angle between incoming light and surface normal
    cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);

    if (etaI == etaT) {
        return 0.0f;
    }

    float eta = etaI / etaT;
    float sin2ThetaI = glm::max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(1.0f - sin2ThetaT);

    float rParallel = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);
    float rPerpendicular = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);

    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

static __forceinline__ __device__
float TrowbridgeReitzD(float NoH, float roughness) { //NoH: surface normal with microfacet normal or halfway direction
    if (NoH <= 0.0f) { //NoH = dot(normal, microfacetNormal)
        return 0.0f;
    }

    float cos2Theta = NoH * NoH;
    float denominator = (1.0f - cos2Theta) + roughness * roughness * cos2Theta;

    return roughness * roughness / (PI * denominator * denominator);
}

static __forceinline__ __device__
float Lambda(float NoV, float roughness) {  //NoV: surface normal with direction previous path vertex
    float cos2Theta = NoV * NoV;
    float sin2Theta = glm::max(0.0f, 1.0f - cos2Theta);
    float tan2Theta = sin2Theta / cos2Theta;

    float alpha2Tan2Theta = roughness * roughness * tan2Theta;

    return (-1.0f + sqrtf(1.0f + alpha2Tan2Theta)) / 2.0f;
}


static __forceinline__ __device__
float TrowbridgeReitzG(float NoV, float NoL, float roughness) { //NoL: surface normal with newlyl sampled outgoing direction
    if (NoV <= 0.0f || NoL <= 0.0f) {
        return 0.0f;
    }

    return 1.0f / (1.0f + Lambda(NoV, roughness) + Lambda(NoL, roughness));
}

//Randomly chooses the normal of a tiny mirror on a rough surface
//note that xi is a random number generator
static __forceinline__ __device__
glm::vec3 Sample_wh(const glm::vec3& nor, const glm::vec2& xi, float roughness) {
    float phi = TWO_PI * xi.y; //chooses a direction around the surface normal

    float tanTheta2 = roughness * roughness * xi.x / (1.0f - xi.x); //chooses how much the microfacet tilts away from the surface normal, following GGX distribution. small roughness -> concnetrates near nor, larger -> spreads them wider

    //spherial-coordinate direction, local
    float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
    float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 wh(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

    //helper axis, checking if helper axis is 
    glm::vec3 directionNotNormal = fabsf(nor.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);

    glm::vec3 tangent = glm::normalize(glm::cross(directionNotNormal, nor));
    glm::vec3 bitangent = glm::cross(nor, tangent);

    //rotate direction to match actual surface
    return glm::normalize(wh.x * tangent + wh.y * bitangent + wh.z * nor);
}

static __forceinline__ __device__
float TrowbridgeReitzPdf(const glm::vec3& nor, const glm::vec3& wh, float roughness) {
    float NoH = glm::clamp(glm::dot(nor, wh), 0.0f, 1.0f);

    return TrowbridgeReitzD(NoH, roughness) * NoH;
}

static __forceinline__ __device__
glm::vec3 fresnel(float cosTheta, const glm::vec3& R, float roughness){
    cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);

    return R + (glm::max(glm::vec3(1.0f - roughness), R) - R) * powf(1.0f - cosTheta, 5.0f);
}

static __forceinline__ __device__
glm::vec3 f_microfacet_refl(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec3& wo, const glm::vec3& wi, float roughness) {
    float cosThetaO = glm::clamp(glm::dot(nor, wo), -1.0f, 1.0f);
    float cosThetaI = glm::clamp(glm::dot(nor, wi), -1.0f, 1.0f);

    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = wo + wi;

    if (glm::dot(wh, wh) == 0.0f) {
        return glm::vec3(0.0f);
    }

    wh = glm::normalize(wh);

    float NoH = glm::clamp(glm::dot(nor, wh), 0.0f, 1.0f);

    glm::vec3 F = fresnel(glm::dot(wo, wh), albedo, roughness);
    float D = TrowbridgeReitzD(NoH, roughness);
    float G = TrowbridgeReitzG(cosThetaO, cosThetaI, roughness);

    return D * G * F / (4.0f * cosThetaI * cosThetaO);
}

static __forceinline__ __device__
glm::vec3 f_diffuse(const glm::vec3& albedo) {
    return albedo / PI;
}

static __forceinline__ __device__
glm::vec3 f_metallic_roughness(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec3& wo,
    const glm::vec3& wi, float roughness, float metallic) {

    glm::vec3 N = nor;

    float NdotV = glm::clamp(glm::dot(N, wo), 0.0f, 1.0f);
    float NdotL = glm::clamp(glm::dot(N, wi), 0.0f, 1.0f);

    if (NdotV <= 0.0f || NdotL <= 0.0f || roughness <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = wo + wi;

    if (glm::dot(wh, wh) == 0.0f) {
        return glm::vec3(0.0f);
    }

    wh = glm::normalize(wh);
    metallic = glm::clamp(metallic, 0.0f, 1.0f);

    glm::vec3 R = glm::mix(glm::vec3(0.04f), albedo, metallic);
    glm::vec3 F = fresnel(glm::dot(wo, wh), R, roughness);

    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0f) - kS;
    kD *= 1.0f - metallic;

    glm::vec3 diffuse = f_diffuse(albedo);

    float NoH = glm::clamp(glm::dot(N, wh), 0.0f, 1.0f);
    float D = TrowbridgeReitzD(NoH, roughness);
    float G = TrowbridgeReitzG(NdotV, NdotL, roughness);

    glm::vec3 specular = D * G * F / (4.0f * NdotV * NdotL);

    return kD * diffuse + specular;
}

static __forceinline__ __device__
glm::vec3 Sample_f_metallic_roughness(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec3& wo, float roughness, float metallic,
    thrust::default_random_engine& rng, glm::vec3& wiW, float& pdf) {

    wiW = glm::vec3(0.0f);
    pdf = 0.0f;

    if (glm::dot(nor, wo) <= 0.0f || roughness <= 0.0f) {
        return glm::vec3(0.0f);
    }

    metallic = glm::clamp(metallic, 0.0f, 1.0f);

    // Sampling probability
    float pSpecular = 0.5f + 0.5f * metallic;

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float choose = glm::min(u01(rng), 0.99999994f);
    glm::vec3 wi;

    if (choose < pSpecular) {
        glm::vec2 xi;
        xi.x = glm::min(u01(rng), 0.99999994f);
        xi.y = u01(rng);

        glm::vec3 wh = Sample_wh(nor, xi, roughness);

        if (glm::dot(wo, wh) <= 0.0f) {
            return glm::vec3(0.0f);
        }
        wi = glm::normalize(glm::reflect(-wo, wh));
    }
    else {
        wi = glm::normalize(raygenSampleHemisphere(nor, rng));
    }

    float cosThetaI = glm::clamp(glm::dot(nor, wi), 0.0f, 1.0f);
    if (cosThetaI <= 0.0f) {
        return glm::vec3(0.0f);
    }

    // Compute both sampling densities for the chosen direction
    glm::vec3 wh = wo + wi;
    if (glm::dot(wh, wh) == 0.0f) {
        return glm::vec3(0.0f);
    }

    wh = glm::normalize(wh);
    float woDotWh = glm::dot(wo, wh);
    if (woDotWh <= 0.0f) {
        return glm::vec3(0.0f);
    }

    float pdfSpecular = TrowbridgeReitzPdf(nor, wh, roughness) / (4.0f * woDotWh);
    float pdfDiffuse = cosThetaI / PI; //cosine weighted sampler, diffuse direction density

    pdf = pSpecular * pdfSpecular + (1.0f - pSpecular) * pdfDiffuse;

    wiW = wi;

    return f_metallic_roughness(albedo, nor, wo, wi, roughness, metallic);
}

static __forceinline__ __device__
glm::vec3 Sample_f_microfacet_refl(const glm::vec3& albedo, const glm::vec3& nor, const glm::vec2& xi, const glm::vec3& wo, float roughness, glm::vec3& wiW, float& pdf) {
    wiW = glm::vec3(0.0f);
    pdf = 0.0f;

    if (glm::dot(nor, wo) <= 0.0f || roughness <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wh = Sample_wh(nor, xi, roughness);

    float woDotWh = glm::dot(wo, wh);
    if (woDotWh <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 wi = glm::normalize(glm::reflect(-wo, wh));

    //Reflection must stay above shading surface
    if (glm::dot(nor, wi) <= 0.0f) {
        return glm::vec3(0.0f);
    }

    pdf = TrowbridgeReitzPdf(nor, wh, roughness) / (4.0f * woDotWh);

    wiW = wi;

    return f_microfacet_refl(albedo, nor, wo, wi, roughness);
}

//from 5610 pbr.frag sampleSphericalMap
static __forceinline__ __device__
glm::vec3 sampleEnvironment(const glm::vec3& direction) {
    if (params.environmentTexture == 0) {
        return glm::vec3(0.0f);
    }

    glm::vec3 dir = glm::normalize(direction);

    float u = atan2f(dir.z, dir.x) / TWO_PI + 0.5f; //horizontal anglea round the environment
    float v = 0.5f - asinf(glm::clamp(dir.y, -1.0f, 1.0f)) / PI; // asinf - elevation, minus makes upward rays sample the top of the unflipped image
    
    float4 texel = tex2D<float4>(params.environmentTexture, u, v);

    return glm::vec3(texel.x, texel.y, texel.z);
}

static __forceinline__ __device__
double environmentPixelSolidAngle(int y) {
    double pi = 3.14159265358979323846;
    double thetaTop = pi * y / params.environmentHeight;
    double thetaBottom = pi * (y + 1) / params.environmentHeight;

    return (2.0 * pi / params.environmentWidth) * (cos(thetaTop) - cos(thetaBottom));
}

static __forceinline__ __device__
glm::vec3 sampleEnvironmentDirection(thrust::default_random_engine& rng, float& pdf) {
    pdf = 0.0f;

    int width = params.environmentWidth;
    int height = params.environmentHeight;

    if (params.environmentCdf == nullptr || width <= 0 || height <= 0) {
        return glm::vec3(0.0f);
    }

    thrust::uniform_real_distribution<double> u01(0.0, 1.0);

    double select = fmin(u01(rng), 0.9999999999999999); //safety clamp to prevent out of bounds indxing errors, 1.0, binary search or mapping math look for upper bound of CDF, which might return pixelCount = indexing error (is largest possible double)
    size_t pixelCount = static_cast<size_t>(width) * height;

    //find i which sufficies cdf[i] <= select < cdf[i + 1]
    size_t low = 0;
    size_t high = pixelCount;

    //binary search pixel, select pixel according to CDF
    while (low + 1 < high) {
        size_t middle = low + (high - low) / 2;

        if (params.environmentCdf[middle] <= select) {
            low = middle;
        }
        else {
            high = middle;
        }
    }
    size_t pixelIndex = low;
    int x = static_cast<int>(pixelIndex % width); //oh no
    int y = static_cast<int>(pixelIndex / width);

    //probability of choosing the entire pixel
    double probability = params.environmentCdf[pixelIndex + 1] - params.environmentCdf[pixelIndex];

    //select a direction inside the pixel's spherical area
    // spherical area covered by that pixel in steradians
    double solidAngle = environmentPixelSolidAngle(y);

    if (!(probability > 0.0) || !(solidAngle > 0.0)) {
        return glm::vec3(0.0f);
    }

    double pi = 3.14159265358979323846;

    double xiU = fmin(u01(rng), 0.9999999999999999); //random u
    double xiV = fmin(u01(rng), 0.9999999999999999); //random v

    //horizontal position
    double u = (x + xiU) / width; // x is the column, xiU chooses fractional position inside it, dividing by width normalizes u
    double phi = (u - 0.5) * (2.0 * pi); //converting u into phi gives the horinzontal angle (-180, 180)

    //vertical position (theta is measured downward from +Y)
    double cosTop = cos(pi * y / height);
    double cosBottom = cos(pi * (y + 1) / height);

    //Uniform solid angle sampling inside selected pixel
    //interpolate between top and bottom boundareies in cos(theta) because uniform horizontal angle and uniform cos(theta) produce uniform spherical area
    double cosTheta = cosTop + xiV * (cosBottom - cosTop);

    //trig identity
    double sinTheta = sqrt(fmax(0.0, 1.0 - cosTheta * cosTheta)); //interpolate using cosTheta rather than theta to sample uniformly in solid angle

    //probability is the whole pixel's probability, so divide it by solidAngle
    pdf = static_cast<float>(probability / solidAngle);

    //spherical coords become a direction
    return glm::normalize(glm::vec3(
        static_cast<float>(sinTheta * cos(phi)), static_cast<float>(cosTheta), 
        static_cast<float>(sinTheta * sin(phi))));
}

//for MIS, environment PDF for directions chosen by BSDF
//what probability density would be assigned to each direction for the environment sampler
//1. Convert the supplied direction to horizontal angle and vertical angle
//2. Find the corresponding HDR pixel
//3. Retrieve that pixel's probability from the CDf
//4. Divide by its solid angle
static __forceinline__ __device__
float environmentPdf(const glm::vec3& direction) {
    int width = params.environmentWidth;
    int height = params.environmentHeight;

    if (params.environmentCdf == nullptr || width <= 0 || height <= 0) {
        return 0.0f;
    }

    double dx = direction.x;
    double dy = direction.y;
    double dz = direction.z;

    if (dx * dx + dy * dy + dz * dz <= 0.0) {
        return 0.0f;
    }

    double pi = 3.14159265358979323846;

    //find corresponding pixel on HDR
    //atan(dz, dx) recovers horizontal angle, divide by 2*pi + 0.5 swithces range to 0 , 1
    double u = atan2(dz, dx) / (2.0 * pi) + 0.5;
    u -= floor(u); //Wrap horizontal seam to [0,1), equivaelnt directions at the two panormala's edges select the same column

    //vertical angle
    //computes angle from +Y, sqrt is the direction's horizontal length
    //sqrt(dx^2, dz^2 = sintheta, dy = costheta
    double theta = atan2(sqrt(dx * dx + dz * dz), dy);

    int x = static_cast<int>(u * width);
    int y = static_cast<int>((theta / pi) * height);

    if (x >= width){
        x = width - 1;
    }
    if (y >= height) {
        y = height - 1;
    }

    //get the probability of selecting the pixel
    size_t pixelIndex = static_cast<size_t>(y) * width + x;
    double probability = params.environmentCdf[pixelIndex + 1] - params.environmentCdf[pixelIndex];

    //divide pixel probability by solid angle
    double solidAngle = environmentPixelSolidAngle(y);

    float pdf;
    if (solidAngle > 0.0f) {
        pdf = static_cast<float>(probability / solidAngle);
    }
    else {
        pdf = 0.0f;
    }

    return pdf;
}

//shadow feeler ray
static __forceinline__ __device__
bool isEnvironmentVisible(const glm::vec3& hitPoint, const glm::vec3& geometricNormal, const glm::vec3& wi) {
    float offsetSign = glm::dot(geometricNormal, wi) >= 0.0f ? 1.0f : -1.0f;
    glm::vec3 origin = hitPoint + offsetSign * 0.001f * geometricNormal;

    unsigned int index = optixGetLaunchIndex().x;

    optixTrace(
        params.gasHandle,
        make_float3(origin.x, origin.y, origin.z),
        make_float3(wi.x, wi.y, wi.z),
        0.001f,
        FLT_MAX,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0);

    return params.intersections[index].t < 0.0f;
}

//MIS powerheuristic
static __forceinline__ __device__
float powerHeuristic(float pdfA, float pdfB) {
    if (!(pdfA > 0.0f)) {
        return 0.0f;
    }
    if (!(pdfB > 0.0f)) {
        return 1.0f;
    }

    float scale = fmaxf(pdfA, pdfB);
    float a = pdfA / scale;
    float b = pdfB / scale;

    return(a * a) / (a * a + b * b);
}

//diffuse/specular mixture due to the random choice implementation
static __forceinline__ __device__
float metallicRoughnessPdf(const glm::vec3& nor, const glm::vec3& wo, const glm::vec3& wi, float roughness, float metallic) {
    float NoV = glm::dot(nor, wo);
    float NoL = glm::dot(nor, wi);

    if (NoV <= 0.0f || NoL <= 0.0f) {
        return 0.0f;
    }

    glm::vec3 wh = wo + wi;

    if (glm::dot(wh, wh) <= 0.0f) {
        return 0.0f;
    }

    wh = glm::normalize(wh);

    const float VoH = glm::dot(wo, wh);
    if (VoH <= 0.0f) {
        return 0.0f;
    }


    const float pSpecular = 0.5f + 0.5f * metallic;
    const float pdfSpecular = TrowbridgeReitzPdf(nor, wh, roughness) / (4.0f * VoH);
    const float pdfDiffuse = NoL / PI;

    return pSpecular * pdfSpecular + (1.0f - pSpecular) * pdfDiffuse;
}

static __forceinline__ __device__
float4 sampleMaterialTexture(int textureId, bool color, const glm::vec2& uv, float4 fallback) {
    if (textureId < 0 || params.textures == nullptr) {
        return fallback;
    }

    //Even slot - color with sRGB decoding, Odd slot - data without sRGB decoding
    size_t slot = 2 * static_cast<size_t>(textureId);

    if (!color) {
        slot += 1;
    }

    cudaTextureObject_t texture = params.textures[slot];

    if (texture == 0) {
        return fallback;
    }

    return tex2D<float4>(texture, uv.x, uv.y);
}

static __forceinline__ __device__
void applyMaterialTextures(Material& material, ShadeableIntersection& hit, const glm::vec3& rayDirection) {
    if (material.emittance > 0.0f) {
        return;
    }

    //Base color
    if (material.baseColorTextureId >= 0) {
        float4 texel = sampleMaterialTexture(material.baseColorTextureId, true, hit.texcoord, make_float4(1, 1, 1, 1));

        material.color *= glm::vec3(texel.x, texel.y, texel.z);
    }

    //Metallic Roughness map
    if (material.useMetallicRoughness != 0 && material.metallicRoughnessTextureId >= 0) {
        float4 texel = sampleMaterialTexture(material.metallicRoughnessTextureId, false, hit.texcoord, make_float4(1, 1, 1, 1));
        
        float textureRoughness = glm::clamp(texel.y, 0.0f, 1.0f);
        float textureMetallic = glm::clamp(texel.z, 0.0f, 1.0f);

        // material.roughness already has roughnessFactor squared
        material.roughness *= textureRoughness * textureRoughness;
        material.metallic *= textureMetallic;
    }

    if (material.useMetallicRoughness == 0 || material.hasRefractive > 0.0f || material.normalTextureId < 0) {
        return;
    }

    // Missing or degenerate tangent frame,  keep the original normal
    if (glm::dot(hit.surfaceTangent, hit.surfaceTangent) <= 0.000000000001f ||
        glm::dot(hit.surfaceBitangent, hit.surfaceBitangent) <= 0.000000000001f) {
        return;
    }

    float4 texel = sampleMaterialTexture(material.normalTextureId, false, hit.texcoord, make_float4(0.5f, 0.5f, 1.0f, 1.0f));

    // Decode RGB from 0 - 1 into a tangent space direction, normalScale changes the strength of the sideways components.
    glm::vec3 localNormal((2.0f * texel.x - 1.0f) * material.normalScale, (2.0f * texel.y - 1.0f) * material.normalScale, 2.0f * texel.z - 1.0f);
    if (glm::dot(localNormal, localNormal) <= 0.000000000001f) {
        return;
    }

    localNormal = glm::normalize(localNormal);

    // Tangent to world space
    glm::vec3 mappedNormal = hit.surfaceTangent * localNormal.x + hit.surfaceBitangent * localNormal.y + hit.surfaceNormal * localNormal.z;
    if (glm::dot(mappedNormal, mappedNormal) <= 0.000000000001f) {
        return;
    }

    mappedNormal = glm::normalize(mappedNormal);
    glm::vec3 wo = -glm::normalize(rayDirection);

    glm::vec3 geometricNormal = glm::normalize(hit.geometricNormal);
    if (glm::dot(geometricNormal, wo) < 0.0f) {
        geometricNormal = -geometricNormal;
    }

    //Only accept a mapped normal facing the geometric hemishpere and wo, safeguarding extreme normal map directions
    if (glm::dot(mappedNormal, geometricNormal) > 0.0f && glm::dot(mappedNormal, wo) > 0.0f) {
        hit.surfaceNormal = mappedNormal;
    }
}

//__forceinline__ tells the compiler to inline the function into the caller rather than use an ordinary function call
// 1: pure refractive
// 2: metal/roughness (when 0, still diffuse + specular (plastic), more metallic means specular color moves towards the base color and removes diffuse)
// 3: pure reflective (+ roughness)
// 4: pure diffuse
static __forceinline__ __device__
void raygenShadeFakeMaterial(PathSegment& path, ShadeableIntersection& intersection, 
    glm::vec3& accumulatedLight, float& previousBsdfPdf, bool& previousEnvironmentMIS) {
    if (intersection.t > 0.0f) {

        previousEnvironmentMIS = false;
        previousBsdfPdf = 0.0f;

        thrust::default_random_engine rng = raygenMakeRandomEngine(params.iteration, path.pixelIndex, path.remainingBounces);

        Material material = params.materials[intersection.materialId];

        //apply textures
        applyMaterialTextures(material, intersection, path.ray.direction);

        glm::vec3 materialColor = material.color;

        //light source
        if (material.emittance > 0.0f) {
            accumulatedLight += path.color * material.color * material.emittance;

            path.remainingBounces = 0;
            return;
        }
        //glass (refraction + fresnel)
        else if (material.hasRefractive > 0.0f) {
            glm::vec3 incident = glm::normalize(path.ray.direction);
            bool frontFace = glm::dot(incident, intersection.geometricNormal) < 0.0f;

            glm::vec3 normal = frontFace ? intersection.geometricNormal : -intersection.geometricNormal;

            float etaI = frontFace ? 1.0f : material.indexOfRefraction;
            float etaT = frontFace ? material.indexOfRefraction : 1.0f;
            float eta = etaI / etaT;

            float cosThetaI = glm::clamp(glm::dot(-incident, normal), 0.0f, 1.0f);
            float F = fresnelDielectric(cosThetaI, etaI, etaT); //probability

            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

            glm::vec3 nextDirection;

            if (F >= 1.0f || u01(rng) < F) { //F becomes the possibility to reflect/refract
                nextDirection = glm::reflect(incident, normal);
            }
            else {
                nextDirection = glm::refract(incident, normal, eta);

                path.color *= eta * eta; //pbrt's radiacne transport correction
            }

            nextDirection = glm::normalize(nextDirection);

            //exact surface position
            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;
            //move onto the side the outgoing ray is traveling toward
            float offsetSign = glm::dot(nextDirection, intersection.geometricNormal) > 0.0f ? 1.0f : -1.0f; //reflection stays in th incident medium, refraction crosses the surface

            path.ray.origin = hitPoint + offsetSign * 0.001f * intersection.geometricNormal;
            path.ray.direction = nextDirection;
            path.remainingBounces--;

            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }

        }
        //metallic 0 - colroed diffuse reflection and neutral specular reflection (plastic)
        //metallic 1 - colored specular reflection, no diffuse component (metal)
        else if (material.useMetallicRoughness != 0) {
            glm::vec3 wo = -glm::normalize(path.ray.direction);

            glm::vec3 geometricNormal = glm::normalize(intersection.geometricNormal);
            if (glm::dot(geometricNormal, wo) < 0.0f) {
                geometricNormal = -geometricNormal;
            }

            glm::vec3 nor = glm::normalize(intersection.surfaceNormal);
            if (glm::dot(nor, geometricNormal) < 0.0f) {
                nor = -nor;
            }
            if (glm::dot(nor, wo) <= 0.0f) {
                nor = geometricNormal;
            }

            float roughness = glm::clamp(material.roughness, 0.001f, 1.0f);
            float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);

            bool useEnvironmentMIS = USE_ENVIRONMENT_MIS != 0 && params.environmentTexture != 0 &&
                params.environmentCdf != nullptr && params.environmentWidth > 0 && params.environmentHeight > 0 &&
                path.remainingBounces > 1;

            if (useEnvironmentMIS) {
                float lightPdf = 0.0f;
                glm::vec3 wiLight = sampleEnvironmentDirection(rng, lightPdf);
                float NoL = glm::dot(nor, wiLight);


                if (lightPdf > 0.0f && NoL > 0.0f && glm::dot(geometricNormal, wiLight) > 0.0f) {
                    glm::vec3 directHitPoint = path.ray.origin + intersection.t * path.ray.direction;

                    if (isEnvironmentVisible(directHitPoint, geometricNormal, wiLight)){
                        glm::vec3 fLight = f_metallic_roughness(material.color, nor, wo, wiLight, roughness, metallic);

                        float bsdfPdf = metallicRoughnessPdf(nor, wo, wiLight, roughness, metallic);

                        float weight = powerHeuristic(lightPdf, bsdfPdf);

                        accumulatedLight += path.color * fLight * sampleEnvironment(wiLight) * (NoL * weight / lightPdf);
                        }
                }
            }

            glm::vec3 wiW;
            float pdf = 0.0f;

            glm::vec3 f = Sample_f_metallic_roughness(material.color, nor, wo, roughness, metallic, rng, wiW, pdf);

            if (!(pdf > 0.0f) || glm::dot(wiW, geometricNormal) <= 0.0f) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
                return;
            }

            float cosThetaI = glm::clamp(glm::dot(nor, wiW), 0.0f, 1.0f);

            previousBsdfPdf = pdf;
            previousEnvironmentMIS = useEnvironmentMIS;

            path.color *= f * cosThetaI / pdf;

            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;

            path.ray.origin = hitPoint + 0.001f * geometricNormal;
            path.ray.direction = glm::normalize(wiW);
            path.remainingBounces--;

            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }
        }
        //reflective + roughness
        else if (material.hasReflective > 0.0f) {
            glm::vec3 incident = glm::normalize(path.ray.direction); //world space ray
            glm::vec3 wo = -incident;

            //If shading normal reflects ray into the object, use geometric normal instead
            //Geometric normal, flip if incident is on the opposite side
            glm::vec3 geometricNormal = glm::normalize(intersection.geometricNormal);

            if (glm::dot(geometricNormal, wo) < 0.0f) {
                geometricNormal = -geometricNormal;
            }

            //Align the shading normal with the geometric normal
            glm::vec3 nor = glm::normalize(intersection.surfaceNormal);
            if (glm::dot(nor, geometricNormal) < 0.0f) {
                nor = -nor;
            }

            if (glm::dot(nor, wo) <= 0.0f) {
                nor = geometricNormal;
            }

            float roughness = glm::clamp(material.roughness, 0.0f, 1.0f);
            glm::vec3 wiW;

            if (roughness == 0.0f) {
                // Perfect mirror
                glm::vec3 wh = nor;
                wiW = glm::reflect(-wo, wh);

                if (glm::dot(wiW, geometricNormal) <= 0.0f){
                    wh = geometricNormal;
                    wiW = glm::reflect(-wo, wh);
                }
                path.color *= fresnel(glm::dot(wo, wh), material.color, 0.0f);
            }
            else {
                //Avoid numerically extreme GGX values
                roughness = glm::max(roughness, 0.001f);

                thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

                glm::vec2 xi;
                xi.x = glm::min(u01(rng), 0.99999994f); //largest representable float below 1.0f, exclusive
                xi.y = glm::min(u01(rng), 0.99999994f);

                float pdf = 0.0f;

                glm::vec3 f = Sample_f_microfacet_refl(material.color, nor, xi, wo, roughness, wiW, pdf);

                if (!(pdf > 0.0f) || glm::dot(wiW, geometricNormal) <= 0.0f) {
                    path.color = glm::vec3(0.0f);
                    path.remainingBounces = 0;
                    return;
                }

                float cosThetaI = glm::clamp(glm::dot(nor, wiW), 0.0f, 1.0f);
                path.color *= f * cosThetaI / pdf;
            }

            glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;

            path.ray.origin = hitPoint + 0.001f * geometricNormal;
            path.ray.direction = glm::normalize(wiW);
            path.remainingBounces--;

            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }
        }
        else {
            glm::vec3 intersectionPoint = getPointOnRay(path.ray, intersection.t);

            // scatterRay, would separating this optimize further?
            glm::vec3 nextDirection = raygenSampleHemisphere(intersection.surfaceNormal, rng);
            path.ray.origin = intersectionPoint;
            path.ray.direction = nextDirection;

            path.color *= material.color;
            path.remainingBounces--;

            // Bounce limit
            if (path.remainingBounces <= 0) {
                path.color = glm::vec3(0.0f);
                path.remainingBounces = 0;
            }
        }
    }
    else {
        float weight = 1.0f;

        if (previousEnvironmentMIS) {
            float lightPdf = environmentPdf(path.ray.direction);
            weight = powerHeuristic(previousBsdfPdf, lightPdf);
        }

        accumulatedLight += path.color * sampleEnvironment(path.ray.direction) * weight;
        //path.color = glm::vec3(0.0f); -> upon miss, now sample environment map
        
        path.remainingBounces = 0;
        return;
    }
}


static __forceinline__ __device__
PathSegment generateRaygenCameraPath(unsigned int index)
{
    RaygenCamera& cam = params.camera;

    int pixelIndex = (int)(index);
    int x = pixelIndex % cam.width;
    int y = pixelIndex / cam.width;

    thrust::default_random_engine rng = raygenMakeRandomEngine(params.iteration, pixelIndex, 0);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float xJitter = u01(rng);
    float yJitter = u01(rng);

    //restriction due to initialization of global __constant__ variable
    const glm::vec3 position(cam.position.x, cam.position.y, cam.position.z);
    const glm::vec3 view(cam.view.x, cam.view.y, cam.view.z);
    const glm::vec3 up(cam.up.x, cam.up.y, cam.up.z);
    const glm::vec3 right(cam.right.x, cam.right.y, cam.right.z);

    PathSegment path{};

    path.ray.origin = position;
    path.ray.direction = glm::normalize(view //normal pinhole camera ray jittered within pixel
        - right * cam.pixelLength.x * ((float)(x + xJitter) - (float)(cam.width) * 0.5f)
        - up * cam.pixelLength.y * ((float)(y + yJitter) - (float)(cam.height) * 0.5f));

    //thin-lens DOF
    if (cam.apertureRadius > 0.0f && cam.focalDistance > 0.0f) {
        //glm::vec3 forward = glm::normalize(view);

        //focal lane is a flat plane perpendicular to the camera's forward dir, at distance focalLength
        //ray direction is usually not parallel to foward, so the distance along the ray to the plane is longer than focal length
        //so project direction onto forward, and divide by cos angle to get distance along the ray
        float tFocus = cam.focalDistance / glm::dot(path.ray.direction, view); //find focus point

        //we know all rays should converge at this point in the focus plane, the pinhole ray just becomes an anchor
        glm::vec3 focusPoint = path.ray.origin + tFocus * path.ray.direction;

        float radius = cam.apertureRadius * sqrtf(u01(rng)); //sample uniform var then multiply by radius to get random point on aperture
        float angle = 6.28318530718f * u01(rng); //radians for polar
        float lensX = radius * cosf(angle);
        float lensY = radius * sinf(angle);

        //consider skipping this
        //glm::vec3 lensRight = glm::normalize(right);
        //glm::vec3 lensUp = glm::normalize(glm::cross(lensRight, forward));

        path.ray.origin = position + lensX * right + lensY * up; //update origin to a point on the physical lens
        path.ray.direction = glm::normalize(focusPoint - path.ray.origin); //update the direction
    }

    path.color = glm::vec3(1.0f);
    path.pixelIndex = pixelIndex;
    path.remainingBounces = params.maxBounces;

    return path;
}

extern "C" __global__ void __raygen__pathtrace()
{
    const unsigned int index = optixGetLaunchIndex().x;

    if (index >= params.numPaths) {
        return;
    }

    // Read once, all bounce updates operate on this local path.
    PathSegment path;

    // toggle
    if (params.generateCameraRays != 0) {
        path = generateRaygenCameraPath(index);
    }
    else {
        path = params.paths[index];
    }

    //when updating directly to path.color, it served 2 roles. 1. during bouncing - throughput(the weight accumulated from material interactions), 2. when reaching a light - the completed lighting contribution
    //this worked only because the path collected light only when it terminated, but in order to add direct sampling we need throughput
    glm::vec3 accumulatedLight(0.0f);

    float previousBsdfPdf = 0.0f;
    bool previousEnvironmentMIS = false;

    while (path.remainingBounces > 0) {
        const float3 origin = make_float3(path.ray.origin.x,path.ray.origin.y,path.ray.origin.z);
        const float3 direction = make_float3(path.ray.direction.x,path.ray.direction.y,path.ray.direction.z);

        optixTrace(
            params.gasHandle,
            origin,
            direction,
            0.001f,
            FLT_MAX,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0);

        // Existing CH/MS programs write this invocation's slot.
        // optixTrace returns after the selected program completes.
        ShadeableIntersection intersection = params.intersections[index];

        raygenShadeFakeMaterial(path, intersection, accumulatedLight, previousBsdfPdf, previousEnvironmentMIS);
    }

    path.color = accumulatedLight;

    // Preserve pixelIndex for the existing finalGather kernel.
    params.outputPaths[index] = path;
}
