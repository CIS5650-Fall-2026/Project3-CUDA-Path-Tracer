#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define COMPACT_TOGGLE 1
#define CONTIGUOUS_TOGGLE 0
#define ANTIALIASING_ON 1
#define LOAD_TRIANGLES 1
#define BASIC_BOUND_CHECK 0
#define BETTER_BOUND_CHECK 1
#define PROCEDURAL_TEXTURE 0
#define CUSTOM_TEXTURE 1

#define SHOW_KERNEL_RUNTIMES 0

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
// TODO: static variables for device memory, any extra info you need, etc
static Triangle* dev_triangles = NULL;
static node* dev_nodes = NULL;
static Texture* dev_textures = NULL;
static unsigned char* dev_textureData = NULL;

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
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_nodes, scene->nodes.size() * sizeof(node));
    cudaMemcpy(dev_nodes, scene->nodes.data(), scene->nodes.size() * sizeof(node), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textureData, scene->textureData.size() * sizeof(unsigned char));
    cudaMemcpy(dev_textureData, scene->textureData.data(), scene->textureData.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_triangles);
    cudaFree(dev_nodes);
    cudaFree(dev_textures);
    cudaFree(dev_textureData);

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

        // TODO: implement antialiasing by jittering the ray
#if ANTIALIASING_ON
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

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
    ShadeableIntersection* intersections,
    Triangle* triangles,
    int triangles_size,
    glm::vec3 meshBoundsMin,
    glm::vec3 meshBoundsMax,
    node* nodes
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hitMaterialId = -1;
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
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hitMaterialId = geom.materialid;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }
        // bound test
        if (LOAD_TRIANGLES && triangles_size > 0) {
            if (BETTER_BOUND_CHECK) {
                glm::vec3 bvh_intersect, bvh_normal;
                glm::vec2 bvh_uv;
                int bvh_materialId = -1;
                float bvh_t = betterMeshIntersectionTest(nodes, triangles, pathSegment.ray, bvh_intersect, bvh_normal, bvh_uv, bvh_materialId);
                if (bvh_t > 0.0f && t_min > bvh_t)
                {
                    t_min = bvh_t;
                    hitMaterialId = bvh_materialId;
                    intersect_point = bvh_intersect;
                    uv = bvh_uv;
                    normal = bvh_normal;
                }
            }
            else if (BASIC_BOUND_CHECK) {
                if (meshBoundIntersectionTest(pathSegment.ray, meshBoundsMin, meshBoundsMax)) {
                    for (int i = 0; i < triangles_size; i++)
                    {
                        Triangle& tri = triangles[i];
                        glm::vec2 tmp_uv;
                        t = triangleIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
                        if (t > 0.0f && t_min > t)
                        {
                            t_min = t;
                            hitMaterialId = tri.materialid;
                            intersect_point = tmp_intersect;
                            uv = tmp_uv;
                            normal = tmp_normal;
                        }
                    }
                }
            }
            else {
                // brute force
                for (int i = 0; i < triangles_size; i++)
                {
                    Triangle& tri = triangles[i];
                    glm::vec2 tmp_uv;
                    t = triangleIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        hitMaterialId = tri.materialid;
                        intersect_point = tmp_intersect;
                        uv = tmp_uv;
                        normal = tmp_normal;
                    }
                }
            }
        }

        if (hitMaterialId == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hitMaterialId;
            intersections[path_index].uv = uv;
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
        ShadeableIntersection intersection = shadeableIntersections[idx];
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
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
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

// copied from shadeFakeMaterial, adjusted for shade real material
__global__ void shadeRealMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Texture* textures, 
    unsigned char* textureData,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            float u = intersection.uv.x - floorf(intersection.uv.x);
            float v = intersection.uv.y - floorf(intersection.uv.y);
            if (material.textureId >= 0) {
#if PROCEDURAL_TEXTURE
                glm::vec3 colorA(1.0f, 0.0f, 0.0f);
                glm::vec3 colorB(0.0f, 1.0f, 0.0f);
                int checkerU = (int)floorf(u * 10.0f);
                int checkerV = (int)floorf(v * 10.0f);
                if ((checkerU + checkerV) % 2 == 0) {
                    materialColor = colorA;
                }
                else {
                    materialColor = colorB;
                }
#else
#if CUSTOM_TEXTURE
                Texture tex = textures[material.textureId];
                int px = min((int)(u * tex.width), tex.width - 1);
                int py = min((int)(v * tex.height), tex.height - 1);
                int idx = tex.offset + (py * tex.width + px) * tex.channels;
                materialColor = glm::vec3(textureData[idx], textureData[idx+1], textureData[idx+2]) / 255.0f;
#else
                materialColor = glm::vec3(1.0f, 1.0f, 1.0f);
#endif
#endif
            }
            material.color = materialColor;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction, intersection.surfaceNormal, material, rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
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

// instead of finalGather this would be inside the while loop so all paths are shown
__global__ void iterationGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment& iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces == 0) {
            image[iterationPath.pixelIndex] += iterationPath.color;
            iterationPath.remainingBounces = -1;
        }
    }
}

/**
 * Helper for thrust ordering
 */
struct materialIdFromShadeableIntersection {
    __host__ __device__ int operator()(const ShadeableIntersection& temp) {
        return temp.materialId;
    }
};

/**
 * Helper for thrust stream compaction
 */
struct terminatePath {
    __host__ __device__ bool operator()(const PathSegment& p) {
        return p.remainingBounces <= 0;
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;

#if SHOW_KERNEL_RUNTIMES
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(start);
#endif
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_triangles,
            hst_scene->triangles.size(),
            hst_scene->meshBoundsMin,
            hst_scene->meshBoundsMax,
            dev_nodes
        );
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float computeIntersectionsMS = 0;
        cudaEventElapsedTime(&computeIntersectionsMS, start, stop);
#endif
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        
        // order dev_paths and dev_intersections
#if CONTIGUOUS_TOGGLE
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(start);
#endif
        thrust::device_vector<int> temp(num_paths);
        thrust::transform(dev_intersections, dev_intersections + num_paths, temp.begin(), materialIdFromShadeableIntersection());
        thrust::sort_by_key(
            temp.begin(), temp.end(),
            thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections))
        );
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float thrustMS = 0;
        cudaEventElapsedTime(&thrustMS, start, stop);
#endif
#endif // CONTIGUOUS_TOGGLE

#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(start);
#endif
        shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_textureData,
            depth
        );
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float shadeRealMaterialMS = 0;
        cudaEventElapsedTime(&shadeRealMaterialMS, start, stop);
#endif
        
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(start);
#endif
        iterationGather<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_image, dev_paths);
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float iterationGatherMS = 0;
        cudaEventElapsedTime(&iterationGatherMS, start, stop);
#endif
        
#if COMPACT_TOGGLE
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(start);
#endif
        PathSegment* newEnd = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, terminatePath());
#if SHOW_KERNEL_RUNTIMES
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float thrustCompactMS = 0;
        cudaEventElapsedTime(&thrustCompactMS, start, stop);
#endif
        num_paths = newEnd - dev_paths;
        iterationComplete = num_paths == 0;
#else // COMPACT_TOGGLE
#if SHOW_KERNEL_RUNTIMES
        float thrustCompactMS = 0;
#endif
        iterationComplete = depth >= traceDepth;
#endif // COMPACT_TOGGLE

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

#if SHOW_KERNEL_RUNTIMES
#if CONTIGUOUS_TOGGLE
        printf("bounce: %d, sum: %f, computeIntersections: %f, shadeRealMaterial: %f, iterationGather: %f, compact: %f, contiguous: %f\n", 
            depth, 
            computeIntersectionsMS + shadeRealMaterialMS + iterationGatherMS + thrustCompactMS + thrustMS,
            computeIntersectionsMS, shadeRealMaterialMS, iterationGatherMS, thrustCompactMS, thrustMS);
#else
        printf("bounce: %d, sum: %f, computeIntersections: %f, shadeRealMaterial: %f, iterationGather: %f, compact: %f\n",
            depth, 
            computeIntersectionsMS + shadeRealMaterialMS + iterationGatherMS + thrustCompactMS,
            computeIntersectionsMS, shadeRealMaterialMS, iterationGatherMS, thrustCompactMS);
#endif
#endif
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
