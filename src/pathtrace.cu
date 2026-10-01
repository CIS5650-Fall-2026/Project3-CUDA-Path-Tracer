#include "pathtrace.h"

#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "denoise.h"

#define ERRORCHECK 1
#define DENOISE_INTERVAL 50

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

__host__ __device__
Sampler makeSampler(int iter, int pixel, int depth, int dimension, bool halton)
{
    Sampler sampler;
    sampler.rng = makeSeededRandomEngine(iter, pixel, depth);
    sampler.index = iter;
    sampler.pixel = pixel;
    sampler.dimension = dimension;
    sampler.halton = halton;
    return sampler;
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
static PathSegment* dev_paths_alt = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_lights = NULL;
static int lightCount = 0;
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
static glm::vec3* dev_texels = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static glm::vec3* dev_denoised = NULL;

using Clock = std::chrono::high_resolution_clock;

static float msSince(Clock::time_point start)
{
    cudaDeviceSynchronize();
    return std::chrono::duration<float, std::milli>(Clock::now() - start).count();
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

    const RenderState& state = scene->state;
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemcpy(dev_image, state.image.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemcpy(dev_albedo, state.albedo.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemcpy(dev_normal, state.normal.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_alt, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    std::vector<int> lights;
    for (int i = 0; i < scene->geoms.size(); i++)
    {
        const Geom& geom = scene->geoms[i];
        if (scene->materials[geom.materialid].emittance > 0.0f && (geom.type == CUBE || geom.type == SPHERE))
        {
            lights.push_back(i);
        }
    }
    lightCount = lights.size();
    cudaMalloc(&dev_lights, lightCount * sizeof(int));
    cudaMemcpy(dev_lights, lights.data(), lightCount * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    scene->bvh.initDevice();

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_texels, scene->texels.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_texels, scene->texels.data(), scene->texels.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    guiData->msTotal = 0.0f;
    guiData->msIntersect = 0.0f;
    guiData->msSort = 0.0f;
    guiData->msShade = 0.0f;
    guiData->msCompact = 0.0f;
    guiData->msDenoise = 0.0f;

    checkCUDAError("pathtraceInit");
}

void pathtraceDownload()
{
    RenderState& state = hst_scene->state;
    cudaMemcpy(state.albedo.data(), dev_albedo, state.albedo.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(state.normal.data(), dev_normal, state.normal.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_paths_alt);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_lights);
    cudaFree(dev_triangles);
    cudaFree(dev_textures);
    cudaFree(dev_texels);
    cudaFree(dev_albedo);
    cudaFree(dev_normal);
    cudaFree(dev_denoised);
    if (hst_scene != NULL)
    {
        hst_scene->bvh.freeDevice();
    }

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, bool antialias, bool halton, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        Sampler sampler = makeSampler(iter, index, 0, 0, halton);

        float px = (float)x;
        float py = (float)y;
        if (antialias)
        {
            px += nextSample(sampler) - 0.5f;
            py += nextSample(sampler) - 0.5f;
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (px - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (py - (float)cam.resolution.y * 0.5f)
        );

        if (cam.lensRadius > 0.0f)
        {
            float radius = cam.lensRadius * sqrt(nextSample(sampler));
            float angle = TWO_PI * nextSample(sampler);
            glm::vec3 focalPoint = cam.position
                + segment.ray.direction * (cam.focalDistance / glm::dot(segment.ray.direction, cam.view));
            segment.ray.origin += (cam.right * cos(angle) + cam.up * sin(angle)) * radius;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.time = nextSample(sampler);
        segment.mediumDistance = 0.0f;
    }
}

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
    BVHNode* bvhNodes,
    bool useBvh,
    bool cullBounds,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }

        float t;
        glm::vec3 intersect_point;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        ShadeableIntersection hit;

        glm::vec3 tmp_intersect;
        ShadeableIntersection tmp{};

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            Ray ray = pathSegment.ray;
            ray.origin -= geom.velocity * pathSegment.time;
            tmp.materialId = geom.materialid;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, ray, tmp_intersect, tmp.surfaceNormal, tmp.outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp.surfaceNormal, tmp.outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, ray, triangles, bvhNodes, useBvh, cullBounds, tmp);
            }
            else
            {
                t = distanceFieldIntersectionTest(geom, ray, tmp_intersect, tmp.surfaceNormal, tmp.outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                hit = tmp;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            hit.t = t_min;
            const Geom& geom = geoms[hit_geom_index];
            if (geom.type == CUBE || geom.type == SPHERE)
            {
                primitiveUV(geom, intersect_point, hit.uv, hit.tangent);
            }
            intersections[path_index] = hit;
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    bool russianRoulette,
    bool directLighting,
    bool halton,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int* lights,
    int lightCount,
    Texture* textures,
    glm::vec3* texels,
    glm::vec3* albedoImage,
    glm::vec3* normalImage,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0.0f)
    {
        path.remainingBounces = 0;
        return;
    }

    Sampler sampler = makeSampler(iter, path.pixelIndex, path.remainingBounces, 5 + (depth - 1) * 8, halton);
    float mediumTravel = FLT_MAX;
    if (path.mediumDistance > 0.0f)
    {
        mediumTravel = -logf(1.0f - nextSample(sampler)) * path.mediumDistance;
    }

    if (mediumTravel < intersection.t)
    {
        path.ray.origin += path.ray.direction * mediumTravel;
        path.ray.direction = calculateRandomDirectionInSphere(sampler);
        path.color *= path.mediumColor;
    }
    else
    {
        Material material = materials[intersection.materialId];
        bool emissive = material.emittance > 0.0f;
        glm::vec3 emission = material.color;
        if (material.emissiveTexture >= 0)
        {
            emission = sampleTexture(textures[material.emissiveTexture], texels, intersection.uv);
            emissive = glm::max(emission.r, glm::max(emission.g, emission.b)) > 0.1f;
        }
        if (emissive)
        {
            if (depth == 1)
            {
                albedoImage[path.pixelIndex] += emission;
                normalImage[path.pixelIndex] += intersection.surfaceNormal;
            }
            path.color *= emission * material.emittance;
            image[path.pixelIndex] += path.color;
            path.remainingBounces = 0;
            return;
        }

        glm::vec3 intersect = getPointOnRay(path.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;
        glm::vec2 uv = intersection.uv * material.textureScale;
        if (material.texture != NO_TEXTURE)
        {
            material.color *= proceduralTexture(material, intersect);
        }
        if (material.colorTexture >= 0)
        {
            material.color *= sampleTexture(textures[material.colorTexture], texels, uv);
        }
        if (material.bumpTexture >= 0)
        {
            normal = bumpNormal(material, textures[material.bumpTexture], texels, uv, normal, intersection.tangent);
        }

        bool diffuse = material.hasReflective == 0.0f && material.hasRefractive == 0.0f
            && material.scatterDistance == 0.0f;
        if (depth == 1)
        {
            albedoImage[path.pixelIndex] += diffuse ? material.color : glm::vec3(1.0f);
            normalImage[path.pixelIndex] += normal;
        }
        if (directLighting && lightCount > 0 && diffuse && path.remainingBounces == 2)
        {
            scatterRayToLight(path, intersect, normal, material, geoms, lights, lightCount, sampler);
        }
        else
        {
            scatterRay(path, intersect, normal, intersection.outside, material, sampler);
        }
    }
    path.remainingBounces--;

    if (russianRoulette && depth > 3)
    {
        float survive = glm::min(glm::max(path.color.r, glm::max(path.color.g, path.color.b)), 1.0f);
        if (nextSample(sampler) >= survive)
        {
            path.remainingBounces = 0;
        }
        else
        {
            path.color /= survive;
        }
    }
}

struct isPathAlive
{
    __host__ __device__ bool operator()(const PathSegment& path)
    {
        return path.remainingBounces > 0;
    }
};

struct compareMaterial
{
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
    {
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
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * A path's result is added to the image when it reaches a light.

    Clock::time_point iterationStart = Clock::now();

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, guiData->antialiasing, guiData->halton, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    guiData->pathCounts.clear();
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        guiData->pathCounts.push_back(num_paths);

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        Clock::time_point start = Clock::now();
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            hst_scene->bvh.dev_nodes,
            guiData->useBvh,
            guiData->cullBounds,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        guiData->msIntersect += msSince(start);
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.

        if (guiData->sortByMaterial)
        {
            start = Clock::now();
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMaterial());
            guiData->msSort += msSince(start);
        }

        start = Clock::now();
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            guiData->russianRoulette,
            guiData->directLighting,
            guiData->halton,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            dev_lights,
            lightCount,
            dev_textures,
            dev_texels,
            dev_albedo,
            dev_normal,
            dev_image
        );
        checkCUDAError("shade");
        guiData->msShade += msSince(start);

        if (guiData->streamCompaction)
        {
            start = Clock::now();
            PathSegment* alive_end = thrust::copy_if(thrust::device, dev_paths, dev_paths + num_paths, dev_paths_alt, isPathAlive());
            num_paths = alive_end - dev_paths_alt;
            std::swap(dev_paths, dev_paths_alt);
            guiData->msCompact += msSince(start);
        }
        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    if (guiData->denoise && (iter == 1 || iter % DENOISE_INTERVAL == 0 || iter == hst_scene->state.iterations))
    {
        Clock::time_point start = Clock::now();
        pathtraceDownload();
        denoise(hst_scene->state, iter);
        cudaMemcpy(dev_denoised, hst_scene->state.denoised.data(),
            pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        guiData->msDenoise += msSince(start);
    }

    // Send results to OpenGL buffer for rendering
    if (guiData->denoise)
    {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, 1, dev_denoised);
    }
    else
    {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    }

    checkCUDAError("pathtrace");
    guiData->msTotal += msSince(iterationStart);
}
