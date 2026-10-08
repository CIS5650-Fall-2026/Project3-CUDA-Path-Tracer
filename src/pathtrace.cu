#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "photons.h"
#include "sdf.h"

#define SORT_BY_MATERIAL 0

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// ACES filmic tonemapping
static __host__ __device__ glm::vec3 acesFilmicTonemap(glm::vec3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return glm::clamp((x * (a * x + glm::vec3(b))) / (x * (c * x + glm::vec3(d)) + glm::vec3(e)), 0.0f, 1.0f);
}

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::vec3 linearColor = glm::max(pix / (float)iter, glm::vec3(0.0f));

        // color correction 
        glm::vec3 tonemapped = acesFilmicTonemap(linearColor);
        const float invGamma = 1.0f / 2.2f;
        glm::vec3 gammaColor = glm::pow(tonemapped, glm::vec3(invGamma));

        glm::ivec3 color;
        color.x = glm::clamp((int)(gammaColor.x * 255.0f), 0, 255);
        color.y = glm::clamp((int)(gammaColor.y * 255.0f), 0, 255);
        color.z = glm::clamp((int)(gammaColor.z * 255.0f), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Geom* dev_emissiveGeoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialIds = NULL;
static float* dev_depth = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;
    setSDFScene(scene->state.sdfScene);

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_emissiveGeoms, scene->emissiveGeoms.size() * sizeof(Geom));
    cudaMemcpy(dev_emissiveGeoms, scene->emissiveGeoms.data(), scene->emissiveGeoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
    cudaMalloc(&dev_depth, pixelcount * sizeof(float));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_emissiveGeoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
	cudaFree(dev_materialIds);
    cudaFree(dev_depth);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rJitter = u01(rng) - .5;
        float uJitter = u01(rng) - .5;
 
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + rJitter)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + uJitter)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.prevSpecular = true; // count direct camera ray as specular for MIS
        segment.seenDiffuse = false;
        segment.dispersed = false;
        segment.channel = 1;
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Geom* emissiveGeoms,
    int emissive_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min;
        int hit_geom_index;

        findClosestIntersection(pathSegment.ray, geoms, geoms_size, t_min, intersect_point, normal, hit_geom_index,
            emissiveGeoms, emissive_size);

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            intersections[path_index].t = t_min;
            const Geom& hitGeom = (hit_geom_index < geoms_size) ? geoms[hit_geom_index] : emissiveGeoms[hit_geom_index - geoms_size];
            intersections[path_index].materialId = hitGeom.materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__host__ __device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    if (f == 0.0f && g == 0.0f) {
        return 0.0f;
    }
    //float balanced = f / (f + g);
    float power = (f * f) / (f * f + g * g);
    return power;
}

// return f * Li * cos / pdf that is MIS-weighted
__device__ glm::vec3 DirectMIS(
    glm::vec3 isectPos,
    glm::vec3 normal,
    glm::vec3 wo,
    const Material& material,
    Geom* geoms,
    int geoms_size,
    Geom* lights,
    int numLights,
    Material* materials,
    thrust::default_random_engine& rng)
{
    if (numLights <= 0) return glm::vec3(0.0f);
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 n = glm::dot(normal, wo) < 0.0f ? -normal : normal;
    glm::vec3 origin = isectPos + 0.001f * n;
    glm::vec3 f = material.color / PI; // diffuse bsdf

    // choose light from emissiveGeoms
    int chosenLightIdx = (int)(u01(rng) * numLights);
    const Geom& light = lights[chosenLightIdx];
    const Material& lightMat = materials[light.materialid];
    glm::vec3 Le = lightMat.color * lightMat.emittance;
    float area = rectArea(light);

    glm::vec3 final = glm::vec3(0.0f);

    // Light sampling
    {
        glm::vec3 lightPos, lightNor; // populate 
        sampleRect(light, u01(rng), u01(rng), lightPos, lightNor);
        glm::vec3 d = lightPos - origin;
        float dist2 = glm::dot(d, d);
        glm::vec3 wi_g = d / sqrtf(dist2);
        float lambert_g = glm::dot(n, wi_g); // cosTheta on surface
        float cosLight = glm::dot(-wi_g, lightNor); // cosTheta on light

        if (lambert_g > 0.0f && cosLight > 0.0f) {
            // solid angle
            // pdf_area * dA/dw
            float pdf_gg = (1.0f / (float)numLights) * dist2 / (cosLight * area); 
            // occlusion
            Ray shadowRay;
            shadowRay.origin = origin;
            shadowRay.direction = wi_g;
            float t; glm::vec3 p, nor; int hitIdx;
            findClosestIntersection(shadowRay, geoms, geoms_size, t, p, nor, hitIdx, lights, numLights);

            if (hitIdx == geoms_size + chosenLightIdx) { // no occlusion! 
                float pdf_fg = lambert_g / PI;
                float w_gg = PowerHeuristic(1, pdf_gg, 1, pdf_fg);
                final += f * Le / pdf_gg * lambert_g * w_gg;
            }
        }
    }

    // BSDF sampling
    {
        glm::vec3 wi_f = calculateRandomDirectionInHemisphere(n, rng);
        float lambert_f = glm::dot(n, wi_f);
        float pdf_ff = lambert_f / PI;

        if (pdf_ff > 0.0f) {
            Ray bsdfRay;
            bsdfRay.origin = origin;
            bsdfRay.direction = wi_f;
            float t; glm::vec3 p, lightNor; int hitIdx;
            findClosestIntersection(bsdfRay, geoms, geoms_size, t, p, lightNor, hitIdx, lights, numLights);

            // only counts if it hits the SAME light chosen above
            if (hitIdx == geoms_size + chosenLightIdx) {
                float cosLight = glm::dot(-wi_f, lightNor);
                if (cosLight > 0.0f) {
                    float pdf_gf = (1.0f / (float)numLights) * t * t / (cosLight * area);
                    float w_ff = PowerHeuristic(1, pdf_ff, 1, pdf_gf);
                    final += f * Le / pdf_ff * lambert_f * w_ff;
                }
            }
        }
    }

    return final;
}

// Full integrator 
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    Geom* lights,
    int numLights,
    glm::vec3* image,
    int traceDepth) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // intersection exists
        {
            if (pathSegment.remainingBounces <= 0) return;
            glm::vec3 isectPos = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
            Material material = materials[intersection.materialId];

            // Case 0: Hit Light source
            if (material.emittance > 0.0f) {
                bool frontFacing = glm::dot(pathSegment.ray.direction, intersection.surfaceNormal) < 0.0f;
                // camera ray or specular chain from camera -> add Le
                // after diffuse -> already counted by DirectMIS
                if (pathSegment.prevSpecular && (!pathSegment.seenDiffuse || !PHOTON_PASS) && frontFacing) {
					glm::vec3 Le = material.color * material.emittance;
					glm::vec3 contribution = pathSegment.color * Le;
#if ARTISTIC
                    if (pathSegment.remainingBounces < traceDepth) {
                        contribution = glm::min(contribution, glm::vec3(SPECULAR_LE_CLAMP));
                    }
#endif
					image[pathSegment.pixelIndex] += contribution;
                }
                pathSegment.remainingBounces = 0;
            }
            else { // Case 1: Hit non-light source
                // make next ray 
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
                thrust::uniform_real_distribution<float> u01(0, 1);

                // LIGHT TRANSPORT EQUATION 
                // f_r(wo, wi) * cosTheta * L_incoming / pdf(wi)
                if (!(material.hasReflective || material.hasRefractive)) { // DIFFUSE
                    // MIS
                    glm::vec3 wo = -pathSegment.ray.direction;
                    glm::vec3 directLight = DirectMIS(isectPos, intersection.surfaceNormal, wo, material,
                        geoms, geoms_size, lights, numLights, materials, rng);
                    image[pathSegment.pixelIndex] += directLight * pathSegment.color; // directLight * throughput 
                    pathSegment.prevSpecular = false;
                    pathSegment.seenDiffuse = true;
                }
                else { // SPECULAR
                    pathSegment.prevSpecular = true;
                    if (material.dispersive && !pathSegment.dispersed) {
#if ARTISTIC
                        pathSegment.channel = (iter + pathSegment.pixelIndex) % 3;
#else
                        float u = u01(rng);
                        pathSegment.channel = (u < 1.0f / 3.0f) ? 0 : (u < 2.0f / 3.0f) ? 1 : 2;
#endif
                        glm::vec3 mask(0.0f);
                        mask[pathSegment.channel] = 3.0f;
                        pathSegment.color *= mask;
                        pathSegment.dispersed = true;
                    }
                }
                
                // find next ray bounce, bsdf, and pdf
                float pdf;
                glm::vec3 bsdf = scatterRay(pathSegment.ray, isectPos, intersection.surfaceNormal, material, pdf,
                    material.indexOfRefraction[pathSegment.channel], rng);


                float cosTheta = glm::abs(glm::dot(intersection.surfaceNormal, pathSegment.ray.direction));
                pathSegment.color *= bsdf * cosTheta / pdf; // throughput 
                pathSegment.remainingBounces -= 1;
            }
        }
        else { // Case 2: no intersection 
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void writeDepth(int n, PathSegment* paths, ShadeableIntersection* intersections, float* depth)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n)
    {
        float t = intersections[index].t;
        depth[paths[index].pixelIndex] = t > 0.0f ? t : FLT_MAX;
    }
}

__global__ void extractMaterialIds(int n, ShadeableIntersection* intersections, int* materialIds) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n)
    {
		materialIds[index] = intersections[index].materialId;
    }
}

// for stream compaction 
// functor to check if a path is terminated
struct isTerminated {
    __host__ __device__ bool operator()(const PathSegment& p) const {
        return (p.remainingBounces <= 0);
    }
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    // set up ray's PathSegment
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // initializations 
    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    bool iterationComplete = false;

    // bounce loop
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // 0. compute intersections -> dev_intersections
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_emissiveGeoms,
            hst_scene->emissiveGeoms.size(),
            dev_intersections
            );
        checkCUDAError("compute intersections");

        if (depth == 0) {
            writeDepth<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_paths, dev_intersections, dev_depth);
            checkCUDAError("write depth");
        }

#if !CAMERA_PASS
        break;
#endif

        // sort intersections by material id
#if SORT_BY_MATERIAL
        extractMaterialIds<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_intersections,
            dev_materialIds
			);
		checkCUDAError("extract material ids");
        thrust::sort_by_key(
            thrust::device,
            dev_materialIds, dev_materialIds + num_paths,
            thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));
        checkCUDAError("sort by material id");
#endif
        
        // 1. add accumulated color + generate new rays 
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_emissiveGeoms,
            hst_scene->emissiveGeoms.size(),
            dev_image,
            traceDepth
			);
		checkCUDAError("shade and bounce");
        cudaDeviceSynchronize();
        
		// stream compaction to remove terminated paths
        dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_path_end, isTerminated());
        num_paths = dev_path_end - dev_paths;

        depth++;
        iterationComplete = (depth >= traceDepth) || (num_paths <= 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    } // end of the bounce loop

#if PHOTON_PASS
    photonMap(hst_scene, hst_scene->state.photonCount, iter,
        dev_geoms, hst_scene->geoms.size(),
        dev_materials, hst_scene->materials.size(),
        cam, dev_image, dev_depth);
#endif

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);


    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

