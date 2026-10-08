#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "sampling.h"
#include "Volume/volume.h"

#include "thrust_usage.h"

#include "Volume/vdb_usage.h"
#include "Volume/vdb_loading.h"

#include "Temperature/temperature.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, int colorCorrection)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::vec3 col = pix / static_cast<float>(iter);

        // Gamma Reinhard
        if (colorCorrection == 1) {
            col = col / (glm::vec3(1.0f) + col);
            col = pow(col, glm::vec3(2.2f));
        }

        glm::ivec3 color;
        color.x = glm::clamp((int)(col.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(col.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(col.z * 255.0), 0, 255);

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
static Volume* dev_volumes = NULL;
static Light* dev_lights = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static void* d_density = NULL;
static void* d_temperature = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

struct SceneDevice {
    Geom* geom;
    Volume* volumes;
    Light* lights;
    void* volumeDensity;
    int num_geoms;
    int num_volumes;
    int num_lights;
};

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    if (hst_scene->vdbFileNameCurrent != "NONE") {
        if (!LoadNVDB(hst_scene->vdbFileNameCurrent, &hst_scene->volumes[0].invTransform, &d_density, "density"))
            std::cerr << "Failed to find density grid in file: " << hst_scene->vdbFileNameCurrent.c_str() << std::endl;
        if (!LoadNVDB(hst_scene->vdbFileNameCurrent, &hst_scene->volumes[0].invTransform, &d_temperature, "temperature"))
            std::cerr << "Failed to find temperature grid in file: " << hst_scene->vdbFileNameCurrent.c_str() << std::endl;
    }

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    if (scene->volumes.size() > 0) {
        cudaMalloc(&dev_volumes, scene->volumes.size() * sizeof(Volume));
        cudaMemcpy(dev_volumes, scene->volumes.data(), scene->volumes.size() * sizeof(Volume), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(d_density);
    cudaFree(d_temperature);
    cudaFree(dev_volumes);
    cudaFree(dev_lights);
    // if(hst_scene && hst_scene->d_grid) cudaFree(hst_scene->d_grid);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, float focalDistance, float holeSize)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Stochastic Sampling
        const int sideCellCount = 4;
        const float cellSize = 1.0f/static_cast<float>(sideCellCount);
        int cell1d = iter % (sideCellCount*sideCellCount);
        glm::ivec2 cell2d = glm::ivec2(cell1d % sideCellCount, cell1d / sideCellCount);
        glm::vec2 cellPos = glm::vec2(cell2d) * cellSize;
        glm::vec2 offsetInCell = cellSize * glm::vec2(u01(rng), u01(rng));
        glm::vec2 jitter = cellPos + offsetInCell;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter.x)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter.y)
        );

        // Depth of Field
        glm::vec3 focalPoint = segment.ray.direction * focalDistance / dot(normalize(cam.view), segment.ray.direction);
        glm::vec3 startPoint = glm::mat2x3(cam.right, cam.up) * holeSize * calculateRandomPositionOnDisk(rng);
        segment.ray.direction = normalize(focalPoint-startPoint);
        segment.ray.origin = cam.position + startPoint;

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        segment.lastBsdfPdf = -1.0f;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int iter,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    VolumeData vd,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);

        int hit_geom_index = -1;
        float t_min = -1.0f;
        glm::vec3 intersect_point = glm::vec3(0.0f); glm::vec3 normal = glm::vec3(0.0f);
        bool hitVolume = false;
        sceneIntersectionTest(geoms, vd, pathSegment.ray, geoms_size, rng, hit_geom_index, t_min, intersect_point, normal, hitVolume);

        intersections[path_index].pathIdx = path_index;
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else if(hitVolume) {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = vd.volumes[hit_geom_index].materialid;
            intersections[path_index].geomId = hit_geom_index;
        } else {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& p = pathSegments[idx];
        Material material = materials[intersection.materialId];

        if (intersection.t <= 0.0f) {
            p.color = glm::vec3(0.0f);
        }
        else {
            float diffuse = 25.0f * max(0.0f, dot(intersection.surfaceNormal, normalize(glm::vec3(1, -2, 0))));
            p.color = diffuse * material.color;
        }
        image[p.pixelIndex] = static_cast<float>(iter)*p.color;
    }
}

#define MIS 1

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geo,
    int num_geoms,
    VolumeData vd,
    Light* lights,
    int num_lights,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& p = pathSegments[intersection.pathIdx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                const Geom& lGeo = geo[intersection.geomId];
                const Light& l = lights[lGeo.lightid];
                p.color *= (materialColor * material.emittance)
#if MIS
                    * (p.lastBsdfPdf < 0.0f ? 1.0f : powerHeuristic(p.lastBsdfPdf,
                        pdfLight( // TODO: make this pdfLightPlane, and then have a pdfLightVolume too when adding direct light volume sampling (kinda far out)
                            p.ray.origin,
                            p.ray.origin + p.ray.direction * intersection.t,
                            normalize(glm::vec3(lGeo.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f))),
                            num_lights,
                            l
                        )));
#else
                    * 1.0f;
#endif

                p.remainingBounces = 0;
                image[p.pixelIndex] += p.color;

            }
            else {
                // Setup
                Ray& ray = p.ray;
                float pdf = 100.0f;
                glm::vec3 wi = glm::vec3(0.0f);
                glm::vec3 colorMult = glm::vec3(0.0f);
                glm::vec3 hitPoint = ray.origin + intersection.t * ray.direction;

                //
#if MIS
                glm::vec3 directContribution = glm::vec3(0.0f);
                sampleRandomLight(geo, num_geoms, vd, lights, num_lights, materials, p.ray.origin + p.ray.direction * intersection.t, intersection.surfaceNormal, -p.ray.direction, materials[intersection.materialId], rng, directContribution);
                image[p.pixelIndex] += p.color * directContribution;
#endif

                if(material.isVolume) {
                    glm::vec3 wi = sampleHenyeyGreensteinDouble(-p.ray.direction, material.g1, material.g2, material.gBlend, vd, rng, pdf);

                    p.ray.origin = hitPoint;
                    p.ray.direction = wi;

                    if(vd.temperature) {
                        // Scale input temperature, convert to rgb, add some contrast and another scale
                        float temperature = vd.temperatureScale * sampleTemperature(hitPoint, vd, vd.volumes[intersection.geomId]);
                        glm::vec3 baseEmission = kelvin_to_rgb(temperature);
                        glm::vec3 emission = pow(baseEmission, glm::vec3(vd.emissionContrast)) * vd.emissionScale;
                        image[p.pixelIndex] += p.color * emission;
                    }

                    colorMult = material.color * powf(vd.scatteringDecay/vd.densityDecay, static_cast<float>(depth));
                } else {
                    // Bounce & Update Throughput
                    glm::vec3 fLambert = glm::vec3(0.0f);
                    sampleBSDF(-ray.direction, intersection.surfaceNormal, material, rng, pdf, fLambert, wi);

                    p.ray.origin = hitPoint + intersection.surfaceNormal * BOUNCE_EPS;
                    p.ray.direction = wi;

                    colorMult = fLambert / max(1e-5f, pdf);
                }

                p.color *= colorMult;
                p.lastBsdfPdf = pdf;
                p.remainingBounces--;
                
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            p.remainingBounces = 0;
            image[p.pixelIndex] += p.color * 0.4f * glm::vec3(1.0f);// glm::vec3(0.5f, 0.4f, 1.2f);
            p.color = glm::vec3(0.0f);
        }
    }
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

#include <iostream>

#define DEBUG_SYNC

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{

    VolumeData vd;
    vd.volumes = dev_volumes;
    vd.num_volumes = hst_scene->volumes.size();
    vd.density = d_density;
    vd.temperature = guiData->UseTemperature ? d_temperature : nullptr;
    vd.bounceDepth = -1;
    vd.gDecay = 1.0f;// 0.75f;
    vd.densityDecay = 1.0f;// 0.75f;
    vd.scatteringDecay = 1.0f;// 0.75f;
    vd.temperatureScale = guiData->TemperatureScale;
    vd.emissionScale = guiData->EmissionScale;
    vd.emissionContrast = guiData->EmissionContrast;
    vd.extinctionMult = guiData->DensityMultiplier;

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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, guiData->MaxDepth, dev_paths, guiData->FocalDistance, guiData->UseDepthOfField ? 0.5f * guiData->FocalDistance/guiData->ApertureFNumber : 0.0f);
#ifdef DEBUG_SYNC
    cudaDeviceSynchronize();
    checkCUDAError("temp");
#endif
    
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        vd.bounceDepth = depth;

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            iter,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            vd,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        if (guiData->RenderType == 0) {
            shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
                iter,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials,
                dev_image
            );
            iterationComplete = true;
        }
        else {
            if (guiData->SortRaysByMaterial) {
                sortIntersections(dev_intersections, dev_intersections + num_paths);
                cudaDeviceSynchronize();
            }

            shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                depth,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials,
                dev_geoms,
                hst_scene->geoms.size(),
                vd,
                dev_lights,
                hst_scene->lights.size(),
                dev_image
                );
#ifdef DEBUG_SYNC
            cudaDeviceSynchronize();
            checkCUDAError("temp");
#endif

            // Stream compaction
            dev_path_end = removeUnusedPaths(dev_paths, dev_path_end);// thrustremove_if(thrust::device, dev_paths, dev_path_end, should_remove_path());
            num_paths = dev_path_end - dev_paths;

            if (num_paths == 0) {
                iterationComplete = true;
            }
            else if (depth >= guiData->MaxDepth) {
                iterationComplete = true;
            }
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        depth++;
    }

    //// Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image, guiData->ColorCorrection);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
