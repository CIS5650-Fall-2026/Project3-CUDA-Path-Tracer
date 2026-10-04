#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "bvh.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "features.h"

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

// Orders intersections by material. Paths ride along as the value array, so
// sorting the keys permutes both and the two stay index-aligned.
struct materialIdLess
{
    __host__ __device__ bool operator()(const ShadeableIntersection& a,
                                        const ShadeableIntersection& b) const
    {
        return a.materialId < b.materialId;
    }
};

// A path stays in the working set while it still has bounces left.
struct pathIsAlive
{
    __host__ __device__ bool operator()(const PathSegment& segment) const
    {
        return segment.remainingBounces > 0;
    }
};

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static DeviceBVH dev_bvh = {};
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// Indices into dev_geoms of every emissive cube, for direct light sampling.
static int* dev_lights = NULL;
static int numLights = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

    bvhUpload(scene->triangles, scene->bvhNodes, dev_bvh);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    std::vector<int> lights;
    for (int i = 0; i < (int)scene->geoms.size(); ++i)
    {
        const Geom& g = scene->geoms[i];
        if (g.type == CUBE && scene->materials[g.materialid].emittance > 0.0f)
        {
            lights.push_back(i);
        }
    }
    numLights = (int)lights.size();
    cudaMalloc(&dev_lights, numLights * sizeof(int));
    cudaMemcpy(dev_lights, lights.data(), numLights * sizeof(int), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    bvhFree(dev_bvh);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_lights);

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

        // Stochastic antialiasing: offset the sample by up to half a pixel in
        // each axis so that averaging over iterations integrates the whole
        // pixel footprint instead of point-sampling its center.
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float jitterX = 0.0f;
        float jitterY = 0.0f;
#if ANTIALIASING
        jitterX = u01(rng) - 0.5f;
        jitterY = u01(rng) - 0.5f;
#endif

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

#if DEPTH_OF_FIELD
        if (cam.lensRadius > 0.0f)
        {
            // Every ray from this pixel, wherever it leaves the lens, passes
            // through the point the pinhole ray hits on the plane of focus. So
            // that plane stays sharp and anything off it spreads over a disk.
            float tFocus = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 focusPoint = cam.position + tFocus * segment.ray.direction;

            // Uniform point on the lens disk (sqrt so samples don't bunch up in
            // the middle).
            float r = cam.lensRadius * sqrtf(u01(rng));
            float theta = TWO_PI * u01(rng);
            segment.ray.origin = cam.position
                + cam.right * (r * cosf(theta))
                + cam.up * (r * sinf(theta));
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }
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
    DeviceBVH bvh,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside = true;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, bvh, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            // TODO: add more intersection tests here... metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
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
            intersections[path_index].outside = outside;
        }
    }
}

// Shade one bounce: accumulate emitted light, or evaluate the BSDF and spawn
// the next ray. A path ends when it hits a light, escapes the scene, or runs
// out of bounces.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int* lights,
    int numLights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces <= 0)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t <= 0.0f)
    {
        // Escaped the scene without finding a light.
        segment.color = BACKGROUND_COLOR;
        segment.remainingBounces = 0;
    }
    else
    {
        Material material = materials[intersection.materialId];

        if (material.emittance > 0.0f)
        {
            // Hit a light: the throughput carried so far becomes the sample.
            segment.color *= material.color * material.emittance;
            segment.remainingBounces = 0;
        }
        else
        {
            thrust::default_random_engine rng =
                makeSeededRandomEngine(iter, segment.pixelIndex, segment.remainingBounces);

            glm::vec3 intersectPoint = getPointOnRay(segment.ray, intersection.t);

#if DIRECT_LIGHTING
            bool isDiffuse = material.hasReflective == 0.0f && material.hasRefractive == 0.0f;
            if (segment.remainingBounces == 2 && isDiffuse && numLights > 0)
            {
                thrust::uniform_real_distribution<float> u01(0, 1);
                const Geom& light = geoms[lights[glm::min((int)(u01(rng) * numLights), numLights - 1)]];

                // Uniform point on the cube's bottom (-y) face.
                glm::vec3 lightPoint = multiplyMV(light.transform, glm::vec4(u01(rng) - 0.5f, -0.5f, u01(rng) - 0.5f, 1.0f));
                glm::vec3 lightNormal = glm::normalize( multiplyMV(light.invTranspose, glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));
                float lightArea = light.scale.x * light.scale.z;

                glm::vec3 normal = intersection.surfaceNormal;
                glm::vec3 toLight = lightPoint - intersectPoint;
                float dist2 = glm::dot(toLight, toLight);
                glm::vec3 wi = toLight / sqrtf(dist2);

                float cosSurf = glm::dot(normal, wi);
                float cosLight = -glm::dot(lightNormal, wi);

                if (cosSurf > 0.0f && cosLight > 0.0f)
                {
                    // Lambertian BRDF * cos / pdf, with the area pdf
                    // 1 / (numLights * lightArea) converted to solid angle.
                    segment.color *= material.color / PI * cosSurf * cosLight * lightArea * (float)numLights / dist2;
                    segment.ray.direction = wi;
                    segment.ray.origin = intersectPoint + normal * EPSILON;
                    segment.remainingBounces--;
                }
                else
                {
                    // The sampled point faces away: no direct light this time.
                    segment.remainingBounces = 0;
                }
            }
            else
#endif
            {
                scatterRay(segment, intersectPoint, intersection.surfaceNormal, intersection.outside, material, rng);
            }

#if RUSSIAN_ROULETTE
            // Past the first few bounces, kill a path with probability equal to
            // how much light it has already lost, and boost the survivors by
            // the same amount so the expected contribution is unchanged. Dim
            // paths stop costing a full set of bounces for almost no light.
            if (depth >= RUSSIAN_ROULETTE_MIN_DEPTH && segment.remainingBounces > 0)
            {
                float survival = glm::min(
                    glm::max(segment.color.r, glm::max(segment.color.g, segment.color.b)), 1.0f);
                thrust::uniform_real_distribution<float> u01(0, 1);
                if (u01(rng) >= survival)
                {
                    segment.remainingBounces = 0;
                }
                else
                {
                    segment.color /= survival;
                }
            }
#endif

            if (segment.remainingBounces <= 0)
            {
                // Never reached a light, so this path contributes nothing.
                segment.color = glm::vec3(0.0f);
            }
        }
    }

    pathSegments[idx] = segment;
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
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_bvh,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if SORT_BY_MATERIAL
        // Group paths by material so neighbouring threads in the shading kernel
        // take the same BSDF branch instead of diverging within a warp.
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
            materialIdLess());
        checkCUDAError("sort by material");
#endif

        // --- Shading Stage ---
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            dev_lights,
            numLights
        );
        checkCUDAError("shade one bounce");

#if STREAM_COMPACTION
        PathSegment* path_end = thrust::partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            pathIsAlive());
        checkCUDAError("stream compaction");
        num_paths = static_cast<int>(path_end - dev_paths);
#endif

#if LOG_BOUNCE_COUNTS
        if (iter == 1)
        {
            printf("depth %d: %d paths remaining\n", depth, num_paths);
        }
#endif

        iterationComplete = (num_paths == 0) || (depth >= traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);
    checkCUDAError("final gather");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
