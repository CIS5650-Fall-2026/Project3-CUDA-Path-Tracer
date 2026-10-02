#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include "../stream_compaction/efficient.h"
#include "geometry.h"
#include "sampling.h"
#include <stdexcept>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <utility>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#undef checkCUDAError
#define checkCUDAError(msg) checkPathtraceCUDAErrorFn(msg, FILENAME, __LINE__)
void checkPathtraceCUDAErrorFn(const char* msg, const char* file, int line)
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
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// Device buffers for sampling, geometry, path queues, and display processing.
#include "pathtraceExtensions.cuh"

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    pathtraceFree();
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // Restore raw accumulation when loading a checkpoint and allocate work buffers.
    const size_t n = pixelcount;
    upload(dev_image, scene->state.image);
    allocate(display, n);
    allocate(bloomA, n);
    allocate(bloomB, n);
    allocate(completedPaths, n);
    allocate(pathsTemp, n);
    allocate(hitsTemp, n);
    allocate(keys, n);
    allocate(order, n);
    allocate(scratch, 3 * n + 1024);
    allocate(triangles, scene->triangles.size());
    allocate(lights, scene->lights.size());
    upload(triangles, scene->triangles);
    upload(lights, scene->lights);
    std::vector<int> bases;
    int dimensions = 8 + 8 * scene->state.traceDepth;
    for (int candidate = 2; int(bases.size()) < dimensions; ++candidate)
    {
        bool prime = true;
        for (int divisor : bases)
        {
            if (divisor * divisor > candidate)
                break;
            if (candidate % divisor == 0)
            {
                prime = false;
                break;
            }
        }
        if (prime)
            bases.push_back(candidate);
    }
    allocate(primes, bases.size());
    upload(primes, bases);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // Release path-tracing work buffers.
    dev_image = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_materials = nullptr;
    dev_intersections = nullptr;
    hst_scene = nullptr;
    release(completedPaths);
    for (auto p : timerEvents)
    {
        cudaEventDestroy(p.start);
        cudaEventDestroy(p.end);
    }
    timerEvents.clear();
    release(display);
    release(bloomA);
    release(bloomB);
    release(triangles);
    release(pathsTemp);
    release(hitsTemp);
    release(keys);
    release(order);
    release(lights);
    release(primes);
    release(scratch);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, RenderOptions o, const int* bases)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment = PathSegment{};
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Jitter the primary ray within the pixel when antialiasing is enabled.
        Sampler s{static_cast<unsigned>(index), static_cast<unsigned>(iter - 1), o.seed, o.halton, bases};
        float sampleX = x + (o.antialiasing ? s.at(0) : 0.5f);
        float sampleY = y + (o.antialiasing ? s.at(1) : 0.5f);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        if (o.depthOfField && cam.apertureRadius > 0)
        {
            glm::vec3 focus = cam.position + segment.ray.direction * (cam.focalDistance / glm::dot(segment.ray.direction, cam.view));
            float radius = cam.apertureRadius * sqrtf(s.at(2)), angle = TWO_PI * s.at(3);
            segment.ray.origin += radius * (cosf(angle) * cam.right + sinf(angle) * cam.up);
            segment.ray.direction = glm::normalize(focus - segment.ray.origin);
        }
        segment.ray.time = o.motionBlur ? cam.shutterOpen + (cam.shutterClose - cam.shutterOpen) * s.at(4) : cam.shutterOpen;

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// Find surface intersections; the shading kernel generates subsequent rays.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections, const Triangle* triangles, bool cull, int* keys)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        intersections[path_index] = ShadeableIntersection{};
        intersections[path_index].t = -1;
        intersections[path_index].materialId = -1;
        keys[path_index] = -1;
        if (pathSegment.remainingBounces <= 0) return;
        ShadeableIntersection candidate{}, nearest{};
        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            t = -1;
            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, candidate);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, candidate);
            }
            // Dispatch other geometry types through the extended intersection helpers.
            else if (geom.type == MESH && intersectGeometry(geom, triangles, pathSegment.ray, cull, t_min, candidate))
            {
                t = candidate.t;
                tmp_intersect = candidate.point;
                tmp_normal = candidate.surfaceNormal;
                outside = candidate.frontFace;
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                nearest = candidate;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index] = nearest;
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].point = intersect_point;
            keys[path_index] = geoms[hit_geom_index].materialid;
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// Diagnostic shader colors surface hits without BSDF evaluation.
// The rendering pipeline uses shadePath for light transport.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // Seed the diagnostic sample from the iteration, pixel, and bounce count.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // Apply random attenuation in this diagnostic shader.
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // randomly attenuate the diagnostic color
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    if (!hst_scene || iter <= 0)
        throw std::runtime_error("Invalid renderer state");
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

    ///////////////////////////////////////////////////////////////////////////

    // Generate primary rays with unit throughput. At each bounce, find
    // intersections, shade the paths, and optionally compact survivors.
    // Accumulate completed path radiance once after the bounce loop.

    // Generate camera paths and trace one sample per pixel.
    const RenderOptions o = hst_scene->state.options;
    lastStats = TraceStats{};
    Timer timer;
    timer.begin();

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, o, primes);
    timer.stop(lastStats.generateMs);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        timer.begin();
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections, triangles, o.meshCulling, keys
        );
        timer.stop(lastStats.intersectMs);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // Group paths by material when enabled, then evaluate the BSDF
        // and generate continuation rays.

        timer.begin();
        if (o.materialSorting)
        {
            // Sort small indices, then gather both buffers. Sorting the enlarged
            // path/intersection tuple exceeds radix-sort shared memory on some GPUs.
            thrust::sequence(thrust::device, order, order + num_paths);
            thrust::sort_by_key(thrust::device, keys, keys + num_paths, order);
            reorderPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, order, dev_paths, pathsTemp, dev_intersections, hitsTemp);
            std::swap(dev_paths, pathsTemp);
            std::swap(dev_intersections, hitsTemp);
        }
        timer.stop(lastStats.sortMs);
        timer.begin();
        shadePath<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials, depth - 1, o, dev_geoms, int(hst_scene->geoms.size()),
            triangles, lights, int(hst_scene->lights.size()), primes, completedPaths
        );
        // Update the live-path count after shading and optional compaction.
        timer.stop(lastStats.shadeMs);
        timer.begin();
        if (o.compaction)
        {
            if (o.sharedCompaction)
            {
                num_paths = StreamCompaction::Efficient::compact(num_paths, pathsTemp, dev_paths, scratch);
                std::swap(dev_paths, pathsTemp);
            }
            else
            {
                auto begin = thrust::device_pointer_cast(dev_paths);
                num_paths = int(thrust::remove_if(thrust::device, begin, begin + num_paths, Dead()) - begin);
            }
        }
        timer.stop(lastStats.compactMs);
        int alive = num_paths;
        if (statistics && !o.compaction)
            alive = num_paths - int(thrust::count_if(thrust::device, dev_paths, dev_paths + num_paths, Dead()));
        if (statistics)
            lastStats.activePaths.push_back(alive);
        iterationComplete = num_paths == 0 || depth >= traceDepth; // Based on compaction and the bounce limit.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    timer.begin();
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, completedPaths);
    timer.stop(lastStats.gatherMs);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    timer.begin();
    pathtraceRefresh(pbo, iter);
    timer.stop(lastStats.postMs);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    timer.resolve();
    checkCUDAError("pathtrace");
}
