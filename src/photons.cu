#include "photons.h"
#include "intersections.h" 
#include "utilities.h"    
#include "interactions.h"

#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cfloat>
#include <vector>



__host__ __device__ inline thrust::default_random_engine makePhotonRandomEngine(int iter, int index, int depth)
{
    const unsigned int kPhotonStream = 9999u;
    unsigned int h = utilhash((1u << 31) | (kPhotonStream << 22) | static_cast<unsigned int>(iter))
        ^ utilhash(static_cast<unsigned int>(index))
        ^ utilhash(static_cast<unsigned int>(depth));
    return thrust::default_random_engine(h);
}


__global__ void kernGeneratePhotons(
    int numPhotons,
    int iter,
    int traceDepth,
    Geom light,
    glm::vec3 photonPower,
    int indexOffset,
    Photon* photons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPhotons)
    {
        return;
    }

    thrust::default_random_engine rng = makePhotonRandomEngine(iter, indexOffset + idx, traceDepth + 1);
    thrust::uniform_real_distribution<float> u01(0, 1);

    // sample a point / nor 
    glm::vec3 pos, nor;
    sampleRect(light, u01(rng), u01(rng), pos, nor);
    glm::vec3 direction = calculateRandomDirectionInHemisphere(nor, rng);

    Photon photon;
    photon.origin = pos;
    photon.direction = direction;
    photon.power = photonPower;
    photon.passedGlass = false;
    photon.dispersed = false;
    photon.channel = 1;
    photon.remainingBounces = traceDepth;

    photons[idx] = photon;
}

__global__ void computePhotonIntersections(
    int numPhotons,
    Photon* photons,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPhotons)
    {
        Photon photon = photons[idx];

        Ray r;
        r.origin = photon.origin;
        r.direction = photon.direction;

        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min;
        int hit_geom_index;

        findClosestIntersection(r, geoms, geoms_size, t_min, intersect_point, normal, hit_geom_index);

        if (hit_geom_index == -1)
        {
            intersections[idx].t = -1.0f;
        }
        else
        {
            intersections[idx].t = t_min;
            intersections[idx].materialId = geoms[hit_geom_index].materialid;
            intersections[idx].surfaceNormal = normal;
        }
    }
}

__device__ glm::vec2 projectToScreen(
    const Camera& cam,
    const glm::vec3& pos)
{
    glm::vec3 viewHat = glm::normalize(cam.view);
    glm::vec3 Q = pos - cam.position;
    float z = glm::dot(Q, viewHat);

    if (z > 0.0f) {
        float qx = glm::dot(Q, cam.right);
        float qy = glm::dot(Q, cam.up);
        float viewLength = glm::length(cam.view);

        float px = cam.resolution.x * 0.5f - qx * viewLength / (z * cam.pixelLength.x);
        float py = cam.resolution.y * 0.5f - qy * viewLength / (z * cam.pixelLength.y);

        int x = static_cast<int>(px);
        int y = static_cast<int>(py);
        return glm::vec2(x, y);
    } else {
		return glm::vec2(-1.0f, -1.0f);
    }
}

__device__ float pixelFootprintArea(
    const Camera& cam,
    const glm::vec3& hitPoint,
    const glm::vec3& normal)
{
	glm::vec3 view = glm::normalize(cam.view);
    
	glm::vec3 q = hitPoint - cam.position;
	glm::vec3 pToCam = -glm::normalize(q);
    float d = glm::length(q);
	

    // A_perp: first compute pixel area perp to view axis 
	// px * py * (z^2 / |f|^2) * cos(alpha)
    // |f| = 1, 
	float z = glm::dot(q, view); // depth along view axis
    float cosAlpha = z / d; // ratio betw pixel ray and view ray
    float A_perp = cam.pixelLength.x * cam.pixelLength.y * z * z * cosAlpha;
	
    // A_surface: project it to actual surface area 
    float cosTheta = glm::max(glm::abs(glm::dot(normal, pToCam)), 1e-4f);
    float A_surface = A_perp / cosTheta; 
    return A_surface;
}

__global__ void evalPhoton(Photon* photons, ShadeableIntersection* intersections, int numPhotons, Material* materials, Camera cam, glm::vec3* image, int iter, const float* camDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPhotons) {
        return;
    }
    Photon photon = photons[idx];
    ShadeableIntersection intersection = intersections[idx];

	Material material = materials[intersection.materialId];

    if (intersection.t > 0.0f) {
        Ray photonRay{ photon.origin, photon.direction };
        glm::vec3 hitPoint = getPointOnRay(photonRay, intersection.t);

        if (material.hasReflective || material.hasRefractive) { // glass, mirror
			photon.passedGlass = true;

            thrust::default_random_engine rng = makePhotonRandomEngine(iter, idx, photon.remainingBounces);

            
            if (material.dispersive && !photon.dispersed) {
                thrust::uniform_real_distribution<float> u01(0, 1);
                float u = u01(rng);
                if (u < 1.0f / 3.0f) { photon.channel = 0; photon.power *= glm::vec3(3, 0, 0); }
                else if (u < 2.0f / 3.0f) { photon.channel = 1; photon.power *= glm::vec3(0, 3, 0); }
                else { photon.channel = 2; photon.power *= glm::vec3(0, 0, 3); }
                photon.dispersed = true;
            }

            float ior = material.indexOfRefraction[photon.channel];

            float pdf;
            glm::vec3 bsdf = scatterRay(photonRay, hitPoint, intersection.surfaceNormal, material, pdf, ior, rng);

            float cosTheta = glm::abs(glm::dot(intersection.surfaceNormal, photonRay.direction));
            photon.power *= bsdf * cosTheta / pdf; // throughput 

            photon.origin = photonRay.origin;
            photon.direction = photonRay.direction;
            photon.remainingBounces -= 1;

            photons[idx] = photon;
        }
        else { // diffuse 
            if (photon.passedGlass) {
                
                glm::vec2 screenPos = projectToScreen(cam, hitPoint);

                if (screenPos.x >= 0 && screenPos.x < cam.resolution.x && screenPos.y >= 0 && screenPos.y < cam.resolution.y) {
                    int pixelIndex = (int)screenPos.y * cam.resolution.x + (int)screenPos.x;
                    float hitDist = glm::length(hitPoint - cam.position);
                    if (hitDist <= camDepth[pixelIndex] * 1.02f) {
                        float footprint = pixelFootprintArea(cam, hitPoint, intersection.surfaceNormal);
                        glm::vec3 contribution = photon.power * (material.color / PI) / footprint;
                        atomicAdd(&image[pixelIndex].x, contribution.x);
                        atomicAdd(&image[pixelIndex].y, contribution.y);
                        atomicAdd(&image[pixelIndex].z, contribution.z);
                    }
                }
            }
            
            photons[idx].remainingBounces = 0;
        }
    } else {
        // Photon did not hit any surface
		photons[idx].remainingBounces = 0; // terminated
    }
}

// for stream compaction 
// functor to check if a photon is terminated
struct isTerminatedPhoton {
    __host__ __device__ bool operator()(const Photon& p) const {
        return (p.remainingBounces <= 0);
    }
};


void photonMap(Scene* scene, int numPhotons, int iter, 
    Geom* geoms, int geomsSize, 
    Material* materials, int materialsSize,
    Camera cam, glm::vec3* image, const float* camDepth) {

    const std::vector<Geom>& lights = scene->emissiveGeoms;
    if (lights.empty()) {
        printf("no photonmap light source\n");
        return;
    }

    // 0.TOTAL FLUX OF LIGHT
    // Phi = PI * A * Le
    std::vector<glm::vec3> lightFlux(lights.size());
    std::vector<float> lightFluxScalar(lights.size());
    float totalFlux = 0.0f;
    for (size_t i = 0; i < lights.size(); i++) {
        const Material& m = scene->materials[lights[i].materialid];
        glm::vec3 Le = m.color * m.emittance;
        lightFlux[i] = PI * rectArea(lights[i]) * Le;
        lightFluxScalar[i] = (lightFlux[i].x + lightFlux[i].y + lightFlux[i].z) / 3.0f;
        totalFlux += lightFluxScalar[i];
    }
    if (totalFlux <= 0.0f) { return; }

    Photon* dev_photons;
    cudaMalloc(&dev_photons, numPhotons * sizeof(Photon));
    ShadeableIntersection* dev_photon_intersections;
    cudaMalloc(&dev_photon_intersections, numPhotons * sizeof(ShadeableIntersection));

    int traceDepth = scene->state.traceDepth;
    const int blockSize = 128;

    // for each light...
    int photonsAssigned = 0;
    for (size_t i = 0; i < lights.size(); i++)
    {
        // 1. GET PHOTON COUNT 
        int count = (i + 1 < lights.size())
            ? static_cast<int>(numPhotons * (lightFluxScalar[i] / totalFlux))
            : (numPhotons - photonsAssigned); // remainder
        if (count <= 0) { continue; }

		// 2. distribute PHOTON POWER of light to each photon
        glm::vec3 photonPower = lightFlux[i] / static_cast<float>(count);
        const int numBlocks = (count + blockSize - 1) / blockSize;

        // 3. get POS and DIRECTIONS 
        kernGeneratePhotons<<<numBlocks, blockSize>>>(
            count, iter, traceDepth, lights[i], photonPower,
            photonsAssigned, dev_photons + photonsAssigned);
        checkCUDAError("kernGeneratePhotons");

        photonsAssigned += count;
    }

    // photon tracing loop
    int depth = 0;
	bool iterationComplete = false;
	Photon* dev_photons_end = dev_photons + numPhotons;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_photon_intersections, 0, numPhotons * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (numPhotons + blockSize - 1) / blockSize;

        // 0. compute intersections -> dev_intersections
        computePhotonIntersections << <numblocksPathSegmentTracing, blockSize >> > (
            numPhotons,
            dev_photons,
            geoms,
            geomsSize,
            dev_photon_intersections
            );
        checkCUDAError("compute intersections");
        evalPhoton<<<numblocksPathSegmentTracing, blockSize>>> (
            dev_photons,
            dev_photon_intersections,
            numPhotons,
            materials,
            cam,
            image,
            iter,
            camDepth
			);

        checkCUDAError("evalPhoton");
        cudaDeviceSynchronize();

        // stream compaction to remove terminated paths
        dev_photons_end = thrust::remove_if(thrust::device, dev_photons, dev_photons_end, isTerminatedPhoton());
		numPhotons = dev_photons_end - dev_photons;

        depth++;
        iterationComplete = (depth >= traceDepth) || (numPhotons <= 0);
    } // end of the bounce loop



    cudaFree(dev_photons);
	cudaFree(dev_photon_intersections);
}

