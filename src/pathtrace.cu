#include "glm/detail/func_geometric.hpp"
#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include "featureToggles.h"

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

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
Geom* dev_lights = NULL; 
int numLights = 0; 

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
    // calculating all the lights in the scene 
    std::vector<Geom> lights; 
    for(const Geom& geom : scene ->geoms) {
        if (scene->materials[geom.materialid].emittance > 0.0f) {
            lights.push_back(geom); 
        }
    }
    numLights = lights.size(); 
    cudaMalloc(&dev_lights, lights.size() * sizeof(Geom)); 
    cudaMemcpy(dev_lights, lights.data(), lights.size() * sizeof(Geom), cudaMemcpyHostToDevice); 


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
    cudaFree(dev_lights); 
    dev_lights = NULL; 

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

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if HALTON_SAMPLING
        // Iteration is already 1-based, so the first Halton sample is not 0.
        segment.sampleIndex = iter;
        segment.dimension = 0;
        segment.scramble = (utilhash((unsigned int)index) & 0x00ffffff) * (1.0f / 16777216.0f);
#else
#if STOCHASTIC_ANTIALIASING || DEPTH_OF_FIELD
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
#endif
#endif

#if STOCHASTIC_ANTIALIASING
#if HALTON_SAMPLING
        float sx = (float)x + sample1D(segment);
        float sy = (float)y + sample1D(segment);
#else
        float sx = (float)x + u01(rng);
        float sy = (float)y + u01(rng);
#endif
#else
        // Pixel center. Still consume the two sample dimensions so later features stay put.
        float sx = (float)x + 0.5f;
        float sy = (float)y + 0.5f;
#if HALTON_SAMPLING
        sample1D(segment);
        sample1D(segment);
#endif
#endif

        glm::vec3 pinholeDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sy - (float)cam.resolution.y * 0.5f)
        );

#if DEPTH_OF_FIELD
        // Thin lens: shift the origin on the aperture and aim at the same focal point.
        const float lensRadius = 0.15f;
        // Plane through the specular sphere at (0, 4, 0). The camera sits at z = 10.5 and looks down -Z.
        const float focalDistance = 10.5f;

#if HALTON_SAMPLING
        float r = lensRadius * sqrtf(sample1D(segment));
        float theta = sample1D(segment) * TWO_PI;
#else
        float r = lensRadius * sqrtf(u01(rng));
        float theta = u01(rng) * TWO_PI;
#endif
        glm::vec3 lensOffset = cam.right * (r * cosf(theta)) + cam.up * (r * sinf(theta));

        float tFocus = focalDistance / glm::dot(pinholeDir, cam.view);
        glm::vec3 focusPoint = cam.position + tFocus * pinholeDir;

        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
#else
        segment.ray.origin = cam.position;
        segment.ray.direction = pinholeDir;
#if HALTON_SAMPLING
        sample1D(segment);
        sample1D(segment);
#endif
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.isDirectRay = 0; 
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
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool hitOutside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            bool outside = true;
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
                hitOutside = outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1; // setting material id to -1 for for misses
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = hitOutside ? 1 : 0;
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
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* lights,
    int numLights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Halton state lives on the path. The thrust engine is the fallback sampler.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
#if !HALTON_SAMPLING
            thrust::uniform_real_distribution<float> u01(0, 1);
#endif

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0; // hit the light, stop the bounce
            }
#if DIRECT_LIGHTING
            // terminating the direct ray if it's blocked from reaching back to the light source
            else if (pathSegments[idx].isDirectRay) {
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0; 
            }
            // checking if a direct ray is applicable
            else if(pathSegments[idx].remainingBounces == 1 && material.hasReflective == 0.0f && material.hasRefractive == 0.0f) {
                // pick a light uniformly 
#if HALTON_SAMPLING
                float uPick = sample1D(pathSegments[idx]);
                float uX = sample1D(pathSegments[idx]);
                float uZ = sample1D(pathSegments[idx]);
#else
                float uPick = u01(rng);
                float uX = u01(rng);
                float uZ = u01(rng);
#endif
                int lightIndex = glm::min((int)(uPick * numLights), numLights - 1);
                Geom light = lights[lightIndex]; 
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t); 

                // getting a point on light
                float x = uX - 0.5f;
                float z = uZ - 0.5f;
                glm::vec3 pLight = multiplyMV(light.transform, glm::vec4(x, -0.5f, z, 1.0f));
                glm::vec3 lightNormal = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(0.0f, -1.0f, 0.0f, 0.0f)));

                glm::vec3 toLight = pLight - intersect; 
                float dist2 = glm::dot(toLight, toLight);
                glm::vec3 wi = dist2 < 1e-6f ? glm::vec3(0.0f) : glm::normalize(toLight);

                float cosSurf = glm::max(0.0f, glm::dot(glm::normalize(intersection.surfaceNormal), wi));
                float cosLight = glm::max(0.0f, glm::dot(lightNormal, -wi)); 

                // A backfacing sample or a zero distance makes the weight 0 or NaN.
                if (cosSurf == 0.0f || cosLight == 0.0f || dist2 < 1e-6f) {
                    pathSegments[idx].color = glm::vec3(0.0f);
                    pathSegments[idx].remainingBounces = 0;
                }
                else {
                    glm::vec3 edgeU = multiplyMV(light.transform, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)); 
                    glm::vec3 edgeV = multiplyMV(light.transform, glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
                    float area = glm::length(glm::cross(edgeU, edgeV));
                    float pdf = (1.0f / numLights) / area; 

                    pathSegments[idx].color *= (material.color / PI) * cosSurf * cosLight / (dist2 * pdf);
                    pathSegments[idx].ray.origin = intersect + wi * 0.001f; 
                    pathSegments[idx].ray.direction = wi;
                    pathSegments[idx].isDirectRay = 1;
                }
            } 
#endif
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                // float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                // pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                // pathSegments[idx].color *= u01(rng); // apply some noise because why not

                // calculating the intersect point
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t); 
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng, intersection.outside != 0);

#if RUSSIAN_ROULETTE
                // depth is 1 on the first hit. The first three bounces stay, since
                // those still carry most of the energy. After that, kill dim paths
                // with probability 1-q and scale survivors by 1/q so the estimate
                // stays unbiased. Direct-lighting rays never reach this branch.
                if (depth > 3 && pathSegments[idx].remainingBounces > 0) {
                    glm::vec3 beta = pathSegments[idx].color;
                    float q = glm::max(beta.x, glm::max(beta.y, beta.z));
                    q = glm::clamp(q, 0.05f, 1.0f);
#if HALTON_SAMPLING
                    float uSurvive = sample1D(pathSegments[idx]);
#else
                    float uSurvive = u01(rng);
#endif
                    if (uSurvive > q) {
                        pathSegments[idx].remainingBounces = 0;
                        pathSegments[idx].color = glm::vec3(0.0f);
                    } else {
                        pathSegments[idx].color /= q;
                    }
                }
#endif
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

    if (index < nPaths && iterationPaths[index].remainingBounces == 0)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// helper to determine if a traced path has been terminated
struct isTerminated {
    __host__ __device__
    bool operator()(const PathSegment& path) {
        return path.remainingBounces == 0;
    }
};

// struct for the thrust::sort_by_key function
struct MaterialIdLess {
    __host__ __device__
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId; 
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

    // One iteration is enough to see how many rays compaction keeps.
    if (iter == 1) {
        printf("COMPACTION bounce %d alive %d\n", depth, num_paths);
        fflush(stdout);
    }

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
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
#if SORT_BY_MATERIAL
        // Part 1.2 - sort by material type
        thrust::sort_by_key(
            thrust::device,
            dev_intersections, dev_intersections + num_paths,
            dev_paths,
            MaterialIdLess());
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

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_lights,
            numLights
        );

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

        // TODO: Stream compact away all of the terminated paths
        // perform check for all remainingBounces are 0 using stream compaction. 
        // Aka: ray hits light, misses, or hit max depth. If so, remove from dev_paths 
        PathSegment* new_end =  thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, isTerminated()); 
        num_paths = new_end - dev_paths; // subtracting pointers of the same types gives the number of elements between them
        iterationComplete = (num_paths == 0);

        if (iter == 1) {
            printf("COMPACTION bounce %d alive %d\n", depth, num_paths);
            fflush(stdout);
        } 

        // iterationComplete = true; // TODO: should be based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
