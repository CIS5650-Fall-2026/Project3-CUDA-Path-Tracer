#include "pathtrace.h"
#include "camera.h"
#include "lighting.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/sort.h>

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
    unsigned int h = utilhash(0x80000000u | (unsigned(depth) << 22) | unsigned(iter)) ^ utilhash(index);
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
static int* dev_lights = NULL;
static int lightCount = 0;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
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

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (!scene->triangles.empty()) {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle),
            cudaMemcpyHostToDevice);
    }

    std::vector<int> lights;
    for (int i = 0; i < scene->geoms.size(); i++) {
        if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f) {
            lights.push_back(i);
        }
    }
    lightCount = int(lights.size());
    if (lightCount > 0) {
        cudaMalloc(&dev_lights, lightCount * sizeof(int));
        cudaMemcpy(dev_lights, lights.data(), lightCount * sizeof(int), cudaMemcpyHostToDevice);
    }
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_triangles);
    cudaFree(dev_lights);
    dev_lights = NULL;
    lightCount = 0;
    dev_triangles = NULL;
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, bool antialiasing,
    PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(traceDepth > 0 ? 1.0f : 0.0f);

        float sampleX = float(x) + 0.5f;
        float sampleY = float(y) + 0.5f;
        if (antialiasing) {
            // Keep camera samples separate from the random numbers used for bounces.
            thrust::default_random_engine rng(utilhash(unsigned(iter)) ^ utilhash(unsigned(index) ^ 0xa511e9b3u));
            thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            sampleX = float(x) + uniform(rng);
            sampleY = float(y) + uniform(rng);
        }
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - float(cam.resolution.x) * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - float(cam.resolution.y) * 0.5f)
        );

        if (cam.apertureRadius > 0.0f) {
            thrust::default_random_engine rng(utilhash(unsigned(iter)) ^ utilhash(unsigned(index) ^ 0x63d83595u));
            thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            float u = uniform(rng);
            float v = uniform(rng);
            segment.ray = sampleLens(cam, segment.ray, u, v);
        }
        segment.useMis = false;
        segment.previousPdf = 0.0f;
        segment.previousHitPoint = cam.position;
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
    const Triangle* triangles,
    bool meshCulling,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        // Misses and finished paths sort together without reading old hit data.
        intersections[path_index].t = -1.0f;
        intersections[path_index].materialId = -1;
        intersections[path_index].geomId = -1;
        intersections[path_index].triangleIndex = -1;
        intersections[path_index].surfaceNormal = glm::vec3(0.0f);
        if (pathSegment.remainingBounces <= 0) {
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        int hitTriangle = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            t = -1.0f;
            int triangleIndex = -1;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH) {
                t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_normal, meshCulling, t_min, &triangleIndex);
                tmp_intersect = getPointOnRay(pathSegment.ray, t);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                hitTriangle = triangleIndex;
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
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].triangleIndex = hitTriangle;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__device__ glm::vec3 sampleDirectLight(const PathSegment& path, glm::vec3 hitPoint,
    glm::vec3 normal, const Material& material, const Geom* geoms, int geomCount,
    const Triangle* triangles, const Material* materials, const int* lights, int numLights,
    bool meshCulling, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    int lightIndex = glm::min(int(uniform(rng) * numLights), numLights - 1);
    const Geom& light = geoms[lights[lightIndex]];
    float choice = uniform(rng);
    float u = uniform(rng);
    float v = uniform(rng);
    LightSample sample = sampleLightSurface(light, triangles, choice, u, v);
    if (sample.areaPdf <= 0.0f) return glm::vec3(0.0f);

    normal = glm::normalize(normal);
    if (glm::dot(normal, path.ray.direction) > 0.0f) normal = -normal;
    glm::vec3 toLight = sample.position - hitPoint;
    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= 1e-8f) return glm::vec3(0.0f);
    glm::vec3 direction = toLight / sqrtf(distanceSquared);
    float surfaceCosine = glm::max(0.0f, glm::dot(normal, direction));
    // Emission is two-sided, matching the existing light-hit shader.
    float lightCosine = fabsf(glm::dot(sample.normal, -direction));
    if (surfaceCosine == 0.0f || lightCosine == 0.0f) return glm::vec3(0.0f);

    Ray shadow;
    shadow.origin = hitPoint + normal * 0.0001f;
    glm::vec3 shadowOffset = sample.position - shadow.origin;
    float shadowDistance = glm::length(shadowOffset);
    if (shadowDistance <= 0.0f) return glm::vec3(0.0f);
    shadow.direction = shadowOffset / shadowDistance;
    // Include the sampled light in visibility tests so its far side cannot shine through itself.
    if (shadowBlocked(shadow, shadowDistance - 0.0002f, geoms, geomCount, triangles, meshCulling)) {
        return glm::vec3(0.0f);
    }

    const Material& emitter = materials[light.materialid];
    float lightPdf = sample.areaPdf * distanceSquared / (float(numLights) * lightCosine);
    float bsdfPdf = surfaceCosine / PI;
    float weight = bsdfPdf / lightPdf * powerWeight(lightPdf, bsdfPdf);
    return path.color * material.color * emitter.color * emitter.emittance * weight;
}
// Each path keeps its throughput until it reaches a light or terminates.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int numPaths,
    ShadeableIntersection* intersections,
    PathSegment* paths,
    Material* materials,
    glm::vec3* image,
    const Geom* geoms, int geomCount, const Triangle* triangles,
    const int* lights, int numLights, bool directLighting, bool meshCulling)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPaths || paths[index].remainingBounces <= 0) {
        return;
    }

    PathSegment& path = paths[index];
    ShadeableIntersection intersection = intersections[index];
    if (intersection.t <= 0.0f) {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return;
    }

    const Material& material = materials[intersection.materialId];
    if (material.emittance > 0.0f) {
        path.color *= material.color * material.emittance;
        // Save the contribution before compaction removes this path.
        float weight = 1.0f;
        if (path.useMis && numLights > 0) {
            glm::vec3 point = getPointOnRay(path.ray, intersection.t);
            glm::vec3 offset = point - path.previousHitPoint;
            float distanceSquared = glm::dot(offset, offset);
            float cosine = fabsf(glm::dot(intersection.surfaceNormal, -glm::normalize(offset)));
            if (cosine > 0.0f && distanceSquared > 0.0f) {
                float areaPdf = lightAreaPdf(geoms[intersection.geomId], triangles, point, intersection.triangleIndex);
                float lightPdf = areaPdf * distanceSquared / (float(numLights) * cosine);
                weight = powerWeight(path.previousPdf, lightPdf);
            }
        }
        image[path.pixelIndex] += path.color * weight;
        path.remainingBounces = 0;
        return;
    }

    path.remainingBounces--;
    if (path.remainingBounces == 0) {
        // A path that never reached a light contributes nothing.
        path.color = glm::vec3(0.0f);
        return;
    }

    glm::vec3 hitPoint = getPointOnRay(path.ray, intersection.t);
    bool diffuse = !material.hasReflective && !material.hasRefractive;
    bool sampleLight = directLighting && numLights > 0 && diffuse;
    if (sampleLight) {
        thrust::default_random_engine lightRng(utilhash(unsigned(iter)) ^ utilhash(unsigned(path.pixelIndex))
            ^ utilhash(unsigned(depth) ^ 0xb5297a4du));
        image[path.pixelIndex] += sampleDirectLight(path, hitPoint, intersection.surfaceNormal, material,
            geoms, geomCount, triangles, materials, lights, numLights, meshCulling, lightRng);
    }
    // Store the competing BSDF sample's PDF for a possible light hit next bounce.
    path.useMis = sampleLight;
    path.previousHitPoint = hitPoint;
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth);
    scatterRay(path, hitPoint, intersection.surfaceNormal, material, rng);
    path.previousPdf = fabsf(glm::dot(glm::normalize(intersection.surfaceNormal), path.ray.direction)) / PI;
}
struct FinishedPath {
    __host__ __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces <= 0;
    }
};

struct CompareMaterial {
    __host__ __device__ bool operator()(const ShadeableIntersection& a,
        const ShadeableIntersection& b) const {
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, hst_scene->state.antialiasing, dev_paths);
    checkCUDAError("generate camera ray");

    RenderState& state = hst_scene->state;
    state.activePaths.clear();
    int numPaths = traceDepth > 0 ? pixelcount : 0;
    int tracedDepth = 0;
    for (int depth = 0; depth < traceDepth && numPaths > 0; depth++) {
        int numBlocks = (numPaths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numBlocks, blockSize1d>>>(
            depth, numPaths, dev_paths, dev_geoms, hst_scene->geoms.size(),
            dev_triangles, state.meshCulling, dev_intersections);
        if (state.materialSorting) {
            // Move each path with its intersection so shading still sees the right hit.
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + numPaths,
                dev_paths, CompareMaterial());
        }
        shadeMaterial<<<numBlocks, blockSize1d>>>(
            iter, depth, numPaths, dev_intersections, dev_paths, dev_materials, dev_image,
            dev_geoms, int(hst_scene->geoms.size()), dev_triangles, dev_lights, lightCount,
            state.directLighting, state.meshCulling);
        checkCUDAError("trace and shade bounce");
        tracedDepth++;

        if (state.streamCompaction) {
            PathSegment* end = thrust::remove_if(thrust::device,
                dev_paths, dev_paths + numPaths, FinishedPath());
            numPaths = int(end - dev_paths);
        }
        if (state.recordPathCounts) {
            int active = numPaths;
            if (!state.streamCompaction) {
                active -= int(thrust::count_if(thrust::device,
                    dev_paths, dev_paths + numPaths, FinishedPath()));
            }
            state.activePaths.push_back(active);
        }
    }

    if (guiData != NULL) {
        guiData->TracedDepth = tracedDepth;
    }
    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
