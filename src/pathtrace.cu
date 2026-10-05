#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
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

static int* dev_material_keys = NULL;
static bool enableStreamCompaction = true;
static bool enableMaterialSorting = false;
static bool enableRefraction = true;

static float apertureRadius = 0.3f;
static float focalDistance = 5.5f;
static bool enableDepthOfField = false;
static bool enableDirectLighting = false;
static bool enableMotionBlur = true;
static bool enableRussianRoulette = false;

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
    cudaMalloc(&dev_material_keys, pixelcount * sizeof(int));

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
    cudaFree(dev_material_keys);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool enableDOF, float apertureRadius, float focalDistance, bool enableMotionBlur) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        // TODO: implement antialiasing by jittering the ray
        segment.color = glm::vec3(1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.lastBounceWasSpecular = 1;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        if (enableMotionBlur)
        {
            segment.time = u01(rng);
        }
        else
        {
            segment.time = 0.0f;
        }

        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        float sampleX = (float)x + jitterX;
        float sampleY = (float)y + jitterY;

        glm::vec3 originalDirection = glm::normalize(cam.view - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f));

        if (!enableDOF)
        {
            segment.ray.origin = cam.position;
            segment.ray.direction = originalDirection;
            return;
        }

        //segment.ray.direction = glm::normalize(cam.view
        //    - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
        //    - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        //);
        float focalT = focalDistance / glm::dot(originalDirection, cam.view);
        glm::vec3 focalPoint = cam.position + originalDirection * focalT;

        float lensRadius = apertureRadius * sqrtf(u01(rng));
        float lensAngle = TWO_PI * u01(rng);

        float lensX = lensRadius * cosf(lensAngle);
        float lensY = lensRadius * sinf(lensAngle);

        glm::vec3 lensOffset = cam.right * lensX + cam.up * lensY;

        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
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

        if (pathSegment.remainingBounces <= 0)
        {
            intersections[path_index].t = -1.0f;
            return;
        }

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

            Ray motionRay = pathSegment.ray;
            motionRay.origin -= geom.velocity * pathSegment.time;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, motionRay, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, motionRay, tmp_intersect, tmp_normal, outside);
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

__global__ void generateMaterialKeys(
    int num_paths,
    ShadeableIntersection* intersections,
    Material* materials,
    int* materialKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_paths)
    {
        return;
    }

    ShadeableIntersection intersection = intersections[idx];

    if (intersection.t <= 0.0f)
    {
        materialKeys[idx] = 4;
        return;
    }

    Material material = materials[intersection.materialId];

    if (material.emittance > 0.0f)
    {
        materialKeys[idx] = 3;
    }
    else if (material.hasRefractive > 0.0f)
    {
        materialKeys[idx] = 2;
    }
    else if (material.hasReflective > 0.0f)
    {
        materialKeys[idx] = 1;
    }
    else
    {
        materialKeys[idx] = 0;
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

__device__ bool visibleToLight(glm::vec3 surfacePoint, glm::vec3 surfaceNormal, glm::vec3 lightPoint, int lightIndex, Geom* geoms, int geomsSize, float time)
{
    const float SHADOW_OFFSET = 0.001f;

    glm::vec3 shadowOrigin = surfacePoint + glm::normalize(surfaceNormal) * SHADOW_OFFSET;
    glm::vec3 toLight = lightPoint - shadowOrigin;
    float lightDistance = glm::length(toLight);
    glm::vec3 lightDirection = glm::normalize(toLight);

    Ray shadowRay;
    shadowRay.origin = shadowOrigin;
    shadowRay.direction = lightDirection;

    for (int i = 0; i < geomsSize; i++)
    {
        if (i == lightIndex)
        {
            continue;
        }

        float hitT = -1.0f;
        glm::vec3 tempPoint;
        glm::vec3 tempNormal;
        bool outside = true;

        Ray motionShadowRay = shadowRay;
        motionShadowRay.origin -= geoms[i].velocity * time;

        if (geoms[i].type == CUBE)
        {
            hitT = boxIntersectionTest(geoms[i], motionShadowRay, tempPoint, tempNormal, outside);
        }
        else if (geoms[i].type == SPHERE)
        {
            hitT = sphereIntersectionTest(geoms[i], motionShadowRay, tempPoint, tempNormal, outside);
        }

        if (hitT > SHADOW_OFFSET && hitT < lightDistance - SHADOW_OFFSET)
        {
            return false;
        }
    }

    return true;
}

__device__ glm::vec3 calculateDirectLighting(glm::vec3 surfacePoint, glm::vec3 surfaceNormal, Material surfaceMaterial, Geom* geoms, int geomsSize, Material* materials, thrust::default_random_engine& rng, float time)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    int lightCount = 0;

    for (int i = 0; i < geomsSize; i++)
    {
        Material material = materials[geoms[i].materialid];

        if (material.emittance > 0.0f && geoms[i].type == CUBE)
        {
            lightCount++;
        }
    }

    if (lightCount == 0)
    {
        return glm::vec3(0.0f);
    }

    int selectedLight = (int)(u01(rng) * lightCount);

    if (selectedLight >= lightCount)
    {
        selectedLight = lightCount - 1;
    }

    int currentLight = 0;
    int lightIndex = -1;

    for (int i = 0; i < geomsSize; i++)
    {
        Material material = materials[geoms[i].materialid];

        if (material.emittance > 0.0f && geoms[i].type == CUBE)
        {
            if (currentLight == selectedLight)
            {
                lightIndex = i;
                break;
            }

            currentLight++;
        }
    }

    if (lightIndex < 0)
    {
        return glm::vec3(0.0f);
    }

    Geom light = geoms[lightIndex];
    Material lightMaterial = materials[light.materialid];

    float randomX = u01(rng) - 0.5f;
    float randomZ = u01(rng) - 0.5f;

    glm::vec3 localLightPoint = glm::vec3(randomX, -0.5f, randomZ);
    glm::vec3 lightPoint = multiplyMV(light.transform, glm::vec4(localLightPoint, 1.0f));
    lightPoint += light.velocity * time;

    glm::vec3 localLightNormal = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 lightNormal = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(localLightNormal, 0.0f)));

    glm::vec3 toLight = lightPoint - surfacePoint;
    float distanceSquared = glm::dot(toLight, toLight);
    glm::vec3 lightDirection = glm::normalize(toLight);

    float surfaceCosine = glm::max(glm::dot(glm::normalize(surfaceNormal), lightDirection), 0.0f);
    float lightCosine = glm::max(glm::dot(lightNormal, -lightDirection), 0.0f);

    if (surfaceCosine <= 0.0f || lightCosine <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    if (!visibleToLight(surfacePoint, surfaceNormal, lightPoint, lightIndex, geoms, geomsSize, time))
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 lightU = multiplyMV(light.transform, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    glm::vec3 lightV = multiplyMV(light.transform, glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    float lightArea = glm::length(glm::cross(lightU, lightV));

    glm::vec3 emittedLight = lightMaterial.color * lightMaterial.emittance;
    glm::vec3 diffuseBRDF = surfaceMaterial.color / PI;

    glm::vec3 directLight = diffuseBRDF * emittedLight;
    directLight *= surfaceCosine;
    directLight *= lightCosine;
    directLight *= lightArea;
    directLight *= (float)lightCount;
    directLight /= distanceSquared;

    return directLight;
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geomsSize,
    glm::vec3* image,
    bool enableDirectLighting,
    bool enableRussianRoulette,
    bool enableRefraction)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_paths)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];

    if (intersection.t <= 0.0f)
    {
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];
    if (!enableRefraction && material.hasRefractive > 0.0f)
    {
        material.hasRefractive = 0.0f;
    }

    if (material.emittance > 0.0f)
    {
        if (!enableDirectLighting || pathSegment.lastBounceWasSpecular)
        {
            image[pathSegment.pixelIndex] += pathSegment.color * material.color * material.emittance;
        }

        pathSegment.remainingBounces = 0;
        return;
    }

    if (pathSegment.remainingBounces <= 0)
    {
        return;
    }

    //glm::vec3 intersectPoint = getPointOnRay(pathSegment.ray, intersection.t);
    glm::vec3 intersectPoint = pathSegment.ray.origin + intersection.t * glm::normalize(pathSegment.ray.direction);

    thrust::default_random_engine rng =
        makeSeededRandomEngine(
            iter,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        );

    bool isSpecular = material.hasReflective > 0.0f || material.hasRefractive > 0.0f;

    if (enableDirectLighting && !isSpecular)
    {
        glm::vec3 directLight = calculateDirectLighting(intersectPoint, intersection.surfaceNormal, material, geoms, geomsSize, materials, rng, pathSegment.time);
        image[pathSegment.pixelIndex] += pathSegment.color * directLight;
    }

    pathSegment.lastBounceWasSpecular = isSpecular ? 1 : 0;

    scatterRay(
        pathSegment,
        intersectPoint,
        intersection.surfaceNormal,
        material,
        rng
    );

    if (enableRussianRoulette && depth >= 3 && pathSegment.remainingBounces > 0)
    {
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        float survivalProbability = glm::max(pathSegment.color.x, glm::max(pathSegment.color.y, pathSegment.color.z));
        survivalProbability = glm::clamp(survivalProbability, 0.1f, 0.95f);

        if (u01(rng) > survivalProbability)
        {
            pathSegment.remainingBounces = 0;
            return;
        }

        pathSegment.color /= survivalProbability;
    }
}

struct PathTerminated
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces <= 0;
    }
};

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

    if (iter == 1)
    {
        std::cout << std::endl;
        std::cout << "===== Benchmark Configuration =====" << std::endl;
        std::cout << "Stream Compaction: " << (enableStreamCompaction ? "ON" : "OFF") << std::endl;
        std::cout << "Material Sorting: " << (enableMaterialSorting ? "ON" : "OFF") << std::endl;
        std::cout << "Refraction: " << (enableRefraction ? "ON" : "OFF") << std::endl;
        std::cout << "Depth of Field: " << (enableDepthOfField ? "ON" : "OFF") << std::endl;
        std::cout << "Direct Lighting: " << (enableDirectLighting ? "ON" : "OFF") << std::endl;
        std::cout << "Motion Blur: " << (enableMotionBlur ? "ON" : "OFF") << std::endl;
        std::cout << "Russian Roulette: " << (enableRussianRoulette ? "ON" : "OFF") << std::endl;
        std::cout << "===================================" << std::endl;
    }

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

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, enableDepthOfField, apertureRadius, focalDistance, enableMotionBlur);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        if (num_paths == 0)
        {
            break;
        }

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
            dev_intersections
        );
        checkCUDAError("trace one bounce");

        if (enableMaterialSorting)
        {
            generateMaterialKeys << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths,
                dev_intersections,
                dev_materials,
                dev_material_keys
                );

            checkCUDAError("generate material keys");

            thrust::device_ptr<int> keyBegin(dev_material_keys);
            thrust::device_ptr<PathSegment> pathBegin(dev_paths);
            thrust::device_ptr<ShadeableIntersection> intersectionBegin(dev_intersections);

            auto zipBegin = thrust::make_zip_iterator(
                thrust::make_tuple(pathBegin, intersectionBegin)
            );

            thrust::sort_by_key(
                thrust::device,
                keyBegin,
                keyBegin + num_paths,
                zipBegin
            );
        }

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_image,
            enableDirectLighting,
            enableRussianRoulette,
            enableRefraction
            );
        
        checkCUDAError("shade one bounce");
        depth++;

        // TODO: should be based off stream compaction results.

        if (enableStreamCompaction)
        {
            thrust::device_ptr<PathSegment> pathBegin(dev_paths);

            thrust::device_ptr<PathSegment> pathEnd = thrust::remove_if(
                thrust::device,
                pathBegin,
                pathBegin + num_paths,
                PathTerminated()
            );

            num_paths = static_cast<int>(pathEnd - pathBegin);
        }

        if (iter == 1 && enableStreamCompaction)
        {
            std::cout << "Bounce " << depth << ": " << num_paths << " active paths" << std::endl;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        iterationComplete = (num_paths == 0 || depth >= traceDepth);
    }

    // Assemble this iteration and apply it to the image
    //dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
