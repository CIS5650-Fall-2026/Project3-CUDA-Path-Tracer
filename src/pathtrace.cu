#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

// TOGGLES FOR FEATURES HERE
#define SORT_RAYS_BY_MATERIAL 0
#define RAY_STREAM_COMPACTION 0
#define ANTI_ALIASING 0
#define USE_DEPTH_OF_FIELD 1
#define USE_BVH_TREE 1



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
        color.x = glm::clamp((int)((pix.x / iter) * 255.0), 0, 255);
        color.y = glm::clamp((int)((pix.y / iter) * 255.0), 0, 255);
        color.z = glm::clamp((int)((pix.z / iter) * 255.0), 0, 255);

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
static glm::vec3* dev_tempImage = NULL;
static BVHNode* dev_bvhNodes = NULL;
static BVHPrimitive* dev_bvhPrimitives = NULL;
static Geom* dev_lights = NULL;
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

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_tempImage, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_tempImage, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_bvhPrimitives, scene->orderedPrims.size() * sizeof(BVHPrimitive));
    cudaMemcpy(dev_bvhPrimitives, scene->orderedPrims.data(), scene->orderedPrims.size() * sizeof(BVHPrimitive), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

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
    cudaFree(dev_tempImage);
    cudaFree(dev_bvhPrimitives);
    cudaFree(dev_bvhNodes);
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
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pdf = 1.0f;

        // TODO: implement antialiasing by jittering the ray
        if (ANTI_ALIASING) {
            int sampleX = (iter % 16) / 4;
            int sampleY = iter % 4;
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, x * iter, y * iter);
            thrust::uniform_real_distribution<float> u01(-0.125, .125);
            float xRandom = u01(rng);
            float yRandom = u01(rng);
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f - ((float)sampleX * cam.pixelLength.x * 0.25f + xRandom))
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f - ((float)sampleY * cam.pixelLength.y * 0.25f + yRandom))
            );
        }
        else {
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void generateDOFJitteredRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, ShadeableIntersection* intersections, float depthOfField)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        ShadeableIntersection curIntersection = intersections[index];
        float jitterRange = 0.0f;

        if (curIntersection.t > 0.0f) {
            float distanceFromDOF = std::abs (depthOfField - glm::length(cam.position - curIntersection.intersectionPoint));

            jitterRange = distanceFromDOF / 2.2f;
        }

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pdf = 1.0f;

        
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x * iter, y * iter);
        thrust::uniform_real_distribution<float> u01(-1.0f * jitterRange, jitterRange);
        float xRandom = u01(rng);
        float yRandom = u01(rng);
        if (ANTI_ALIASING) {
            int sampleX = (iter % 16) / 4;
            int sampleY = iter % 4;
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f - ((float)sampleX * cam.pixelLength.x * 0.25f + xRandom))
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f - ((float)sampleY * cam.pixelLength.y * 0.25f + yRandom))
            );
        }
        else {
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f +xRandom)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + yRandom)
            );
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
    ShadeableIntersection* intersections,
    BVHNode* bvhNodes,
    BVHPrimitive* bvhPrims,
    int prims_size)
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
        int hitMaterialId = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        bool hitsBoundingVolume = false;
        if (prims_size > 0)
        {
            // check if ray hits any triangles
            int boundingBoxT = boxIntersectionTest(bvhNodes[0].boundingBox, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            hitsBoundingVolume = boundingBoxT > 0.0f;
            //bool hitsBoundingVolume = hitsBoundingBox(bvhNodes[0].boundingBox, pathSegment.ray);
        }

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
            else if (!USE_BVH_TREE && geom.type == TRIANGLE)
            {
                t =  triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                //t = boxIntersectionTest(bvhNodes[0].boundingBox, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else {
                t = -1.0f;
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                hitMaterialId = geoms[hit_geom_index].materialid;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        // just show triangles from bvhPrims
        //for (int i = 0; i < prims_size; i++)
        //{
        //    BVHPrimitive& prim = bvhPrims[i];
        //    Geom triangle = geoms[bvhPrims[i].leafGeomIndex];
        //    t = triangleIntersectionTest(triangle, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        //    // Compute the minimum t from the intersection tests to determine what
        //    // scene geometry object was hit first.
        //    if (t > 0.0f && t_min > t)
        //    {
        //        t_min = t;
        //        hit_geom_index = i;
        //        hitMaterialId = triangle.materialid;
        //        intersect_point = tmp_intersect;
        //        normal = tmp_normal;
        //    }
        //}


        // Intersection test for BVH tree
        if(USE_BVH_TREE && prims_size > 0)
        {
            bool hit = false;
            // maybe don't need below
            glm::vec3 invDir = glm::vec3(1.0f / pathSegment.ray.direction.x, 1.0f / pathSegment.ray.direction.y, 1.0f / pathSegment.ray.direction.z);
            bool dirIsNeg[3] = { invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f };
            int nodesToVist[64];
            int toVistOffset = 0;
            int currentNodeIndex = 0;
            while (true) {
                BVHNode curNode = bvhNodes[currentNodeIndex];
                // if ray intersects with current node
                if (boxIntersectionTest(curNode.boundingBox, pathSegment.ray, tmp_intersect, tmp_normal, outside) > 0.0f) {
                    // if leaf node
                    if (curNode.numPrims > 0) {
                        for (int i = 0; i < curNode.numPrims; ++i) {
                            Geom triangle = geoms[bvhPrims[curNode.firstPrimOffset + i].leafGeomIndex];
                            t = triangleIntersectionTest(triangle, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                            if (t > 0.0f) {
                                hit = true;
                                if (t > 0.0f && t_min > t)
                                {
                                    t_min = t;
                                    hit_geom_index = i;
                                    hitMaterialId = triangle.materialid;
                                    intersect_point = tmp_intersect;
                                    normal = tmp_normal;
                                }
                            }
                        }
                        if (toVistOffset == 0) {
                            break;
                        }
                        currentNodeIndex = nodesToVist[toVistOffset];
                        toVistOffset--;
                    }
                    else {
                        toVistOffset++;
                        nodesToVist[toVistOffset] = curNode.bvhNodeChildIndex_Second;
                        currentNodeIndex = curNode.bvhNodeChildIndex_First;
                    }
                }
                // otherwise visit other nodes in nodesToVist or break
                else {
                    if (toVistOffset == 0) {
                        break;
                    }
                    currentNodeIndex = nodesToVist[toVistOffset];
                    toVistOffset--;
                }
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
            intersections[path_index].materialId = hitMaterialId;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].intersectionPoint = intersect_point;
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
__global__ void shadeMaterials_RandomSample(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color = pathSegments[idx].throughput * (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                // calculate bounced ray dir
                scatterRay(
                    pathSegments[idx],
                    intersection.intersectionPoint,
                    intersection.surfaceNormal,
                    materials[intersection.materialId],
                    rng);
                pathSegments[idx].remainingBounces--;

                // if pdf is 0 or less (outside probable ray bounce directions) terminate ray
                if (pathSegments[idx].pdf <= EPSILON)
                {
                    pathSegments[idx].remainingBounces = 0;
                }
                else {
                    float lightTerm = glm::abs(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction));
                    pathSegments[idx].throughput *= (pathSegments[idx].color*lightTerm) / pathSegments[idx].pdf;
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
        }
    }
}

__global__ void shadeNormalsMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;


            pathSegments[idx].color = intersection.surfaceNormal;
            

        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void shadeMaterialColor(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;


            pathSegments[idx].color = materialColor;


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

__global__ void tempGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] = iterationPath.color;
    }
}

__global__ void finalGatherFromTempImage(int numPixels, glm::vec3* image, glm::vec3* tempImage)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numPixels)
    {
        image[index] += tempImage[index];
    }
}

struct is_terminated
{
    __host__ __device__
        bool operator()(const PathSegment x)
    {
        return x.remainingBounces <= 0;
    }
};

struct material_compare
{
    __host__ __device__
        bool operator()(const ShadeableIntersection left, const ShadeableIntersection right)
    {
        return left.materialId > right.materialId;
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

    float depthOfField = 9.0f;

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

    if (USE_DEPTH_OF_FIELD) {
        // compute intersections to get distance from camera
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            0,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_bvhNodes,
            dev_bvhPrimitives,
            hst_scene->orderedPrims.size()
            );

        // calculate distance from dof and how blurry it should be
        generateDOFJitteredRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, dev_intersections, depthOfField);
    }

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    while (depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing

        numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_bvhNodes,
            dev_bvhPrimitives,
            hst_scene->orderedPrims.size()
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        // Sort by material
        if (SORT_RAYS_BY_MATERIAL) {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_compare());
        }
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterials_RandomSample<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        /*shadeMaterials_DirectLight << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            dev_lights,
            hst_scene->lights.size());*/

        /*shadeNormalsMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );*/

        /*shadeMaterialColor << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );*/

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
        cudaDeviceSynchronize();

        if (RAY_STREAM_COMPACTION)
        {
            dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
            tempGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_tempImage, dev_paths);

            // Stream Compact Terminated Paths
            auto new_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, is_terminated());
            num_paths = new_end - dev_paths;

            cudaDeviceSynchronize();
        }

    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    if (RAY_STREAM_COMPACTION) {
        finalGatherFromTempImage << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_tempImage);
    }
    else {
        finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
