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
#include <algorithm>

#define ERRORCHECK 1


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
static void checkCUDAErrorFn(const char* msg, const char* file, int line)
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
static Triangle* dev_triangles = NULL;
static BoundingVolumeHierarchyNode* dev_boundingVolNodes = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_path_segments = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static bool sortByMaterial = false;
static bool russianRouletteEnabled = false;
//cull any rays that msiss bounding box
static bool boundsCullingEnabled = true;
static bool boundingVolumeHierarchyEnabled = true;

void setMeshBoundingVolHier(bool enabled) {
    boundingVolumeHierarchyEnabled = enabled;
}
bool getMeshBoundingVolHier() {
    return boundingVolumeHierarchyEnabled;
}
void setMeshBoundsCulling(bool enabled) {
    boundsCullingEnabled = enabled;
}

bool getBoundsCullingEnabled() {
    return boundsCullingEnabled;
}
void setRussianRoulettePathTerm(bool enabled) {
    russianRouletteEnabled = enabled;
}
void setMaterialSort(bool enabled) {
    sortByMaterial = enabled;
}

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

    cudaMalloc(&dev_path_segments, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    if (!scene->triangles.empty())
    {
        size_t triBytes = scene->triangles.size() * sizeof(Triangle);
        cudaMalloc(&dev_triangles, triBytes);
        cudaMemcpy(dev_triangles, scene->triangles.data(), triBytes, cudaMemcpyHostToDevice);

        size_t bvhBytes = scene->bvhNodes.size() * sizeof(BoundingVolumeHierarchyNode);
        cudaMalloc(&dev_boundingVolNodes, bvhBytes);
        cudaMemcpy(dev_boundingVolNodes, scene->bvhNodes.data(), bvhBytes, cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_path_segments);
    cudaFree(dev_geoms);
    cudaFree(dev_boundingVolNodes);
    cudaFree(dev_triangles);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 sampleDisc(glm::vec2 u)
{
    glm::vec2 offset = 2.0f * u - glm::vec2(1.0f);
    if (offset.x == 0.0f && offset.y == 0.0f) {
        return glm::vec2(0.0f);
    }
    float theta;
    float offsetDiff = glm::abs(offset.x) - glm::abs(offset.y);
    if (offsetDiff > 0) {
        theta = (PI * offset.y) / (4.0f * offset.x);
        return offset.x * glm::vec2(glm::cos(theta), glm::sin(theta));
    }
    //else
    theta = (PI / 2.0f) - (PI * offset.x) / (4.0f * offset.y);
    return offset.y * glm::vec2(glm::cos(theta), glm::sin(theta));
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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        // offset pixel by random offset value
        float sampledX = (float)x + u01(rng);
        float sampledY = (float)y + u01(rng);

        // Antialiasing using sampled X and Y coordinates
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampledX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampledY - (float)cam.resolution.y * 0.5f)
        );

        //generate ray from random point on lens
        //BLOOPER focal point calculated from lens instead of camera
        // if (cam.radius > 0.0f)
        // {
        //     thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        //     thrust::uniform_real_distribution<float> u01(0, 1);
        //     glm::vec2 randPointOnLens = cam.radius * sampleDisc(glm::vec2(u01(rng), u01(rng)));
        //     segment.ray.origin = cam.position + cam.right * randPointOnLens.x + cam.up * randPointOnLens.y;
        //     glm::vec3 focalPoint = segment.ray.origin + (cam.focalDist / glm::dot(segment.ray.direction, cam.view)) * segment.ray.direction;
        //     segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        // }

               //generate ray from random point on lens
        if (cam.radius > 0.0f) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);
            glm::vec2 randPointOnLens = cam.radius * sampleDisc(glm::vec2(u01(rng), u01(rng)));
            glm::vec3 focalPoint = cam.position + (cam.focalDist / glm::dot(segment.ray.direction, cam.view)) * segment.ray.direction;
            segment.ray.origin = cam.position + cam.right * randPointOnLens.x + cam.up * randPointOnLens.y;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        }

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
    Triangle* triangles,
    BoundingVolumeHierarchyNode* bvhNodes,
    bool useMeshBoundsCulling,
    bool useMeshBVH,
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

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++) {
            Geom& geom = geoms[i];

            if (geom.type == CUBE) {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH) {
                t = meshIntersectionTest(geom, triangles, bvhNodes, pathSegment.ray, useMeshBoundsCulling, useMeshBVH,
                    tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t) {
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

// Real shader: evaluates the BSDF and spawns the next ray for each live path.
// Paths terminate (remainingBounces = 0) when they hit a light, miss the
// scene, or run out of bounces. Only paths that end on a light keep color.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    bool russianRouletteEnabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces <= 0) return; // already terminated

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

        Material material = materials[intersection.materialId];

        if (material.emittance > 0.0f) {
            // Hit a light: add its emission and end the path.
            segment.color *= (material.color * material.emittance);
            segment.remainingBounces = 0;
        }
        else {
            glm::vec3 intersectPoint = segment.ray.origin + segment.ray.direction * intersection.t;
            scatterRay(segment, intersectPoint, intersection.surfaceNormal, material, rng);
            --segment.remainingBounces;
            if (russianRouletteEnabled && depth > 3 && segment.remainingBounces > 0) {
                thrust::uniform_real_distribution<float> u01(0, 1);
                float maxComponent = glm::max(segment.color.x, glm::max(segment.color.y, segment.color.z));
                float q = glm::max(0.05f, 1.0f - maxComponent);
                if (u01(rng) < q) {
                    segment.remainingBounces = 0;
                    segment.color = glm::vec3(0.0f);
                }
                else {
                    segment.color /= (1.0f - q);
                }
            }
            if (segment.remainingBounces == 0) {
                segment.color = glm::vec3(0.0f);
            }
        }
    }
    else {
        // No intersection
        segment.color = BACKGROUND_COLOR;
        segment.remainingBounces = 0;
    }
    pathSegments[idx] = segment;
}

// Orders intersections/paths by material id.
struct materialIdComparator {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};
// true while a path still has bounces left.
struct pathAliveCheck {
    __host__ __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces > 0;
    }
};


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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_path_segments);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_path_segments + pixelcount;
    int num_paths = dev_path_end - dev_path_segments;
    const int total_paths = num_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_path_segments,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_boundingVolNodes,
            boundsCullingEnabled,
            boundingVolumeHierarchyEnabled,
            dev_intersections
        );
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

        if (sortByMaterial) {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths,
                dev_path_segments, materialIdComparator());
        }
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_path_segments,
            dev_materials,
            russianRouletteEnabled
        );
        checkCUDAError("shade");

    //stream compaction
        PathSegment* new_end = thrust::partition(thrust::device, dev_path_segments, dev_path_segments + num_paths, pathAliveCheck());
        num_paths = new_end - dev_path_segments;
        checkCUDAError("stream compaction");


        iterationComplete = (depth >= traceDepth); // TODO: should be based off stream compaction results.
        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(total_paths, dev_image, dev_path_segments);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
