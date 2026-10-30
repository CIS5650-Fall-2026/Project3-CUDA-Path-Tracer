#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "optixSetup.h"
#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cstdlib>
#include <climits>

#define ERRORCHECK 0
#define USE_OPTIX 1
#define TEST_OPTIX_NORMALS 0
#define TEST_MATERIALS 0
#define USE_PARTITION 0 // 0 no partition, 1 thrust, 2 cub, 3 fixed-size launches, no count readback
//Bulk moving things into raygen
#define USE_RAYGEN_LOOP 1
#define USE_RAYGEN_CAMERA 1

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

struct not_terminated {
    __host__ __device__
        bool operator()(PathSegment p) const {
        return p.remainingBounces != 0;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resolution.x || y >= resolution.y || iter <= 0) {
        return;
    }

    int index = x + y * resolution.x;

    //Average the accumulated linear radiance
    glm::vec3 color = image[index] / static_cast<float>(iter);

    //Display exposure for testing
    float exposure = 1.0f;
    color *= exposure;

    color = glm::max(color, glm::vec3(0.0f));

    //Reinhard
    color = color / (glm::vec3(1.0f) + color);

    //Gamma
    color.r = powf(color.r, 1.0f / 2.2f);
    color.g = powf(color.g, 1.0f / 2.2f);
    color.b = powf(color.b, 1.0f / 2.2f);

    color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));

    pbo[index] = make_uchar4(
        (unsigned char)(color.r * 255.0f + 0.5f),
        (unsigned char)(color.g * 255.0f + 0.5f),
        (unsigned char)(color.b * 255.0f + 0.5f),
        255);
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

#if USE_PARTITION == 2 || USE_PARTITION == 3
static PathSegment* dev_partitionOutput = nullptr; //reordered paths
static void* dev_partitionTemp = nullptr; //scrtach workspace used internally by cub
static size_t partitionTempBytes = 0;
static int* dev_partitionCount = nullptr; //holds the number of surviving paths

static void checkCubCuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

static void initCubPartition(int maxPaths) {
    const size_t pathBytes = static_cast<size_t>(maxPaths) * sizeof(PathSegment);
    cudaError_t result = cudaMalloc(reinterpret_cast<void**>(&dev_partitionOutput), pathBytes);
    checkCubCuda(result, "Alloctae CUB partition output");

    result = cudaMalloc(reinterpret_cast<void**>(&dev_partitionCount), sizeof(int));
    checkCubCuda(result, "Allocate CUB partition count");

    // No partition executed with null temporary storage, used to check how much storage we need
    partitionTempBytes = 0;
    result = cub::DevicePartition::If(
        nullptr, //don't partition, only query size
        partitionTempBytes, //CUB writes the required bytes
        dev_paths, 
        dev_partitionOutput, 
        dev_partitionCount, 
        maxPaths, //number of paths we need to support
        not_terminated());
    checkCubCuda(result, "Query CUB partition storage");

    // edge case when partitionTempBytes is 0
    if (partitionTempBytes == 0) {
        partitionTempBytes = 1;
    }

    // allocate amount of needed bytes
    result = cudaMalloc(&dev_partitionTemp, partitionTempBytes);
    checkCubCuda(result, "Allocate CUB partition storage");
}

static int partitionPathsCub(int numPaths) {
    if (numPaths <= 0) {
        return 0;
    }

    // Pass a local copy to use as a CUB parameter
    size_t availableBytes = partitionTempBytes;

    cudaError_t result = cub::DevicePartition::If(
        dev_partitionTemp,
        availableBytes,
        dev_paths,
        dev_partitionOutput,
        dev_partitionCount,
        numPaths,
        not_terminated()
    );
    checkCubCuda(result, "Execute CUB partition");


    //Copy the entire input range for the bounce, surviving paths first
    result = cudaMemcpyAsync(dev_paths, dev_partitionOutput, static_cast<size_t>(numPaths) * sizeof(PathSegment), cudaMemcpyDeviceToDevice, 0);
    checkCubCuda(result, "Copy CUB partition output");

    // CPU needs this for the next bounce
    int activeCount = 0;
    result = cudaMemcpy(&activeCount, dev_partitionCount, sizeof(int), cudaMemcpyDeviceToHost);
    checkCubCuda(result, "Read CUB active-path count");

    return activeCount;
}

static void partitionPathsCubFixed(int pixelcount)
{
    if (pixelcount <= 0) {
        return;
    }

    size_t availableBytes = partitionTempBytes;

    // Partition every path, including previously terminated paths.
    cudaError_t result = cub::DevicePartition::If(
        dev_partitionTemp,
        availableBytes,
        dev_paths,
        dev_partitionOutput,
        dev_partitionCount,
        pixelcount,
        not_terminated()
    );
    checkCubCuda(result, "Execute fixed-size CUB partition");

    // Preserve every path and its final contribution.
    result = cudaMemcpyAsync(dev_paths, dev_partitionOutput, static_cast<size_t>(pixelcount) * sizeof(PathSegment), cudaMemcpyDeviceToDevice, 0);
    checkCubCuda(result, "Copy fixed-size CUB partition output");

    // No GPU-to-CPU survivor-count copy.
}

static void freeCubPartition()
{
    cudaError_t result = cudaFree(dev_partitionOutput);
    checkCubCuda(result, "Free CUB partition output");
    dev_partitionOutput = nullptr;

    result = cudaFree(dev_partitionTemp);
    checkCubCuda(result, "Free CUB partition storage");
    dev_partitionTemp = nullptr;
    partitionTempBytes = 0;

    result = cudaFree(dev_partitionCount);
    checkCubCuda(result, "Free CUB partition count");
    dev_partitionCount = nullptr;
}

#endif

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
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

    // TODO: initialize any extra device memeory you need

#if USE_PARTITION == 2 || USE_PARTITION == 3
    initCubPartition(pixelcount);
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceReset(Scene* scene) {
    if (dev_image == nullptr) {
        pathtraceInit(scene);
        return;
    }

    Camera& cam = scene->state.camera;
    size_t pixelcount = static_cast<size_t>(cam.resolution.x) * static_cast<size_t>(cam.resolution.y);
    cudaError_t result = cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    if (result != cudaSuccess) {
        fprintf(stderr, "Reset accumulation: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }

}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    dev_image = nullptr;
    dev_paths = nullptr;
    dev_geoms = nullptr;
    dev_materials = nullptr;
    dev_intersections = nullptr;
    
#if USE_PARTITION == 2 || USE_PARTITION == 3
    freeCubPartition();
#endif

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
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0., 1.);

        float xJitter = u01(rng);
        float yJitter = u01(rng);

        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)(x + xJitter) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + yJitter) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        if (pathSegments[path_index].remainingBounces <= 0) {
            return;
        }

        PathSegment pathSegment = pathSegments[path_index];

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

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
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
        //skip before reading an intersection that may be from an earlier bounce
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegments[idx].pixelIndex, pathSegments[idx].remainingBounces);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                return;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 intersectPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRay(pathSegments[idx], intersectPoint, intersection.surfaceNormal, material, rng);

                // if the path reaches the bounce limit without hitting a light, turn color to zero
                if (pathSegments[idx].remainingBounces <= 0) {
                    pathSegments[idx].color = glm::vec3(0.f);
                    pathSegments[idx].remainingBounces = 0;
                }
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
            return;
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

//Reads OptiX results and writes a normal based color into each path
__global__ void shadeOptixNormals(int numPaths, const ShadeableIntersection* intersections, PathSegment* paths) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numPaths) {
        return;
    }

    const ShadeableIntersection& hit = intersections[index];
    
    if (hit.t > 0.0f) {
        // Map normal components from [-1,1] into display colors [0,1]
        paths[index].color = 0.5f * (hit.surfaceNormal + glm::vec3(1.0));
    }
    else {
        //Rays that miss the triangle produce a black background
        paths[index].color = glm::vec3(0.0f);
    }
    
    // This diagnostic finishes after the first intersection
    paths[index].remainingBounces = 0;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtraceReadback() {
    const size_t imageBytes = hst_scene->state.image.size() * sizeof(glm::vec3);

    cudaError_t result = cudaMemcpy(hst_scene->state.image.data(), dev_image, imageBytes, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        fprintf(stderr, "Image readback failed: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
    
}

void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    //const Camera& cam = hst_scene->state.camera;
    //Make a local copy so the diagnostic doesn't modify the scene camera
    Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

#if !USE_RAYGEN_CAMERA || !USE_RAYGEN_LOOP
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
#endif
        int depth = 0;
        int num_paths = pixelcount;
        int original_num_paths = pixelcount;
        //PathSegment* dev_path_end = dev_paths + pixelcount;
        //int num_paths = dev_path_end - dev_paths;
        //int original_num_paths = num_paths;

        // --- PathSegment Tracing Stage ---
        // Shoot ray into scene, bounce between objects, push shading chunks

        //NOOOOOPE
#if USE_RAYGEN_LOOP
        launchOptixPaths(
            dev_paths,
            dev_intersections,
            dev_materials,
            pixelcount,
            iter,
            cam,
            traceDepth,
            USE_RAYGEN_CAMERA != 0);

        if (guiData != nullptr) {
            // Exact maximum depth is not read back
            guiData->TracedDepth = -1;
        }
#else

#if USE_PARTITION == 1|| USE_PARTITION == 2
        while(num_paths > 0)
#else
        while (depth < traceDepth)
#endif
        {
            int pathsBeforeBounce = num_paths;

            dim3 numblocksPathSegmentTracing((num_paths + blockSize1d - 1) / blockSize1d);

#if USE_OPTIX
            launchOptixIntersections(dev_paths,dev_intersections,num_paths);
#else
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
                );
#endif
            depth++;

            shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials
                );
#if USE_PARTITION == 1
            //Trhust manages temporary storage internally
            auto start_0s = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, not_terminated());
            num_paths = static_cast<int>(start_0s - dev_paths);
#elif USE_PARTITION == 2
            //We allocate storage once and resuse it
            num_paths = partitionPathsCub(num_paths);
#elif USE_PARTITION == 3
            if (depth < traceDepth) {
                partitionPathsCubFixed(pixelcount);
            }
#endif
            if (iter == 1) {
#if USE_PARTITION == 1 || USE_PARTITION == 2
                std::cout << "Pass " << depth << ": traced " << pathsBeforeBounce << ", surviving " << num_paths << std::endl;
#else
                //std::cout << "Pass " << depth << ": launched " << num_paths << " slots; terminated paths skipped" << std::endl;
#endif
            }
            if (guiData != nullptr) {
                guiData->TracedDepth = depth;
            }
        }

//NOOOOPE
#endif
        // Assemble this iteration and apply it to the image
        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather << <numBlocksPixels, blockSize1d >> > (original_num_paths, dev_image, dev_paths);
        checkCUDAError("gather paths");
        ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    //cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}