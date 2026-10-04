#include "pathtrace.h"
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 0

// Feature Toggles
#define STOCHASTIC_AA 1

#define SORT_BY_MATERIAL 0 // Toggle to 1 to enable sorting

// Sorting predicate
struct CompareByMaterial {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

// Stream compaction predicate
struct PathActive
{
    __host__ __device__ bool operator()(const PathSegment &p) const
    {
        return p.remainingBounces > 0;
    }
};

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line)
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
    thrust::default_random_engine
    makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter, glm::vec3 *image)
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

static Scene *hst_scene = NULL;
static GuiDataContainer *guiData = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static Triangle *dev_triangles = NULL;
static cudaEvent_t intersectionStartEvent = nullptr;
static cudaEvent_t intersectionStopEvent = nullptr;
static BVHNode* dev_bvhNodes = NULL;
int meshAccelMode = 2;   // default to BVH

#define BENCHMARK_INTERSECT 1

void InitDataContainer(GuiDataContainer *imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene *scene)
{
    hst_scene = scene;
    cudaEventCreate(&intersectionStartEvent);
    cudaEventCreate(&intersectionStopEvent);

    const Camera &cam = hst_scene->state.camera;
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
    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    if (!scene->bvhNodes.empty()) {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(),
            scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image); // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_triangles);
    if (intersectionStartEvent != nullptr)
    {
        cudaEventDestroy(intersectionStartEvent);
        intersectionStartEvent = nullptr;
    }
    if (intersectionStopEvent != nullptr)
    {
        cudaEventDestroy(intersectionStopEvent);
        intersectionStopEvent = nullptr;
    }
    cudaFree(dev_bvhNodes);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        int index = x + (y * cam.resolution.x);
        PathSegment &segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        float jitterX = 0.0f;
        float jitterY = 0.0f;

#if STOCHASTIC_AA
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
        jitterX = u01(rng);
        jitterY = u01(rng);
#endif

        segment.ray.direction = glm::normalize(cam.view - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f));

        /*
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
        */

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
    PathSegment *pathSegments,
    Geom *geoms,
    int geoms_size,
    Triangle *triangles,
    BVHNode* bvhNodes, 
    int accelMode,
    ShadeableIntersection *intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t = -1.0f;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool hit_outside = true;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom &geom = geoms[i];
            t = -1.0f;                       // <-- reset every iteration
            outside = true;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, pathSegment.ray, triangles, bvhNodes, accelMode, tmp_intersect, tmp_normal, outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                hit_outside = outside;
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
            intersections[path_index].outside = hit_outside;
        }
    }
}

// Add shadeSimpleMesh above shadeFakeMaterial / shadeMaterial
__global__ void shadeSimpleMesh(
    int iter,
    int num_paths,
    ShadeableIntersection *intersections,
    PathSegment *pathSegments)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = intersections[idx];
        if (intersection.t > 0.0f)
        {
            // Visualize surface normals as RGB colors
            pathSegments[idx].color = glm::abs(intersection.surfaceNormal);
        }
        else
        {
            // Background / missed rays
            pathSegments[idx].color = glm::vec3(0.0f);
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
    ShadeableIntersection *shadeableIntersections,
    PathSegment *pathSegments,
    Material *materials)
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
            if (material.emittance > 0.0f)
            {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else
            {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else
        {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection *shadeableIntersections,
    PathSegment *pathSegments,
    Material *materials,
    glm::vec3 *image) // Added dev_image parameter
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
        return;

    PathSegment &pathSegment = pathSegments[idx];
    if (pathSegment.remainingBounces <= 0)
        return;

    ShadeableIntersection intersection = shadeableIntersections[idx];

    // Ray missed geometry (hit environment / background)
    if (intersection.t <= 0.0f)
    {
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];

    // Hit Light Source (Emissive)
    if (material.emittance > 0.0f)
    {
        // Accumulate light directly into image buffer at the ray's original pixelIndex
        image[pathSegment.pixelIndex] += pathSegment.color * (material.color * material.emittance);
        pathSegment.remainingBounces = 0; // Terminate ray
        return;
    }

    // Hit Diffuse Surface
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
    glm::vec3 intersectPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

    scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, intersection.outside, material, rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image, PathSegment *iterationPaths)
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
void pathtrace(uchar4 *pbo, int frame, int iter)
{
    // Fallback to depth 8 if state traceDepth is uninitialized or 0
    const int traceDepth = (hst_scene->state.traceDepth > 0) ? hst_scene->state.traceDepth : 8;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;
    float iterationIntersectionTimeMs = 0.0f;

    //if (iter == 1 && depth == 0) printf("accel mode = %d\n", meshAccelMode);

    while (num_paths > 0 && depth < traceDepth)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        cudaEventRecord(intersectionStartEvent);
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_triangles,
            dev_bvhNodes,
            meshAccelMode,
            dev_intersections);
        cudaEventRecord(intersectionStopEvent);
        cudaEventSynchronize(intersectionStopEvent);
        float bounceIntersectionTimeMs = 0.0f;
        cudaEventElapsedTime(&bounceIntersectionTimeMs, intersectionStartEvent, intersectionStopEvent);
        iterationIntersectionTimeMs += bounceIntersectionTimeMs;
        checkCUDAError("trace one bounce");

#if SORT_BY_MATERIAL
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
            CompareByMaterial());
#endif

        // Perform BSDF shading and direct radiance accumulation
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_image);
        checkCUDAError("shade material");

        // Stream compaction: keep only active paths (remainingBounces > 0)
        PathSegment *new_end = thrust::partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            PathActive());
        num_paths = new_end - dev_paths;

        depth++;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Send accumulated image results directly to OpenGL PBO
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    if (guiData != NULL)
    {
        ++guiData->IntersectionTimingSamples;
        guiData->IntersectionTimeMs += (iterationIntersectionTimeMs - guiData->IntersectionTimeMs) / guiData->IntersectionTimingSamples;
    }

    // Retrieve image from GPU for host saving
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
