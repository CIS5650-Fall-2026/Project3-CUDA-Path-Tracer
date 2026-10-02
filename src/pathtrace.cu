#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/count.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "mesh.h"

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
    unsigned int h = utilhash((1u << 31) | (static_cast<unsigned int>(depth) << 22) |
        static_cast<unsigned int>(iter)) ^ utilhash(static_cast<unsigned int>(index));
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
        glm::vec3 pix = glm::max(image[index] / static_cast<float>(iter),
            glm::vec3(0.0f));
        pix = pix / (glm::vec3(1.0f) + pix);
        pix = glm::pow(pix, glm::vec3(1.0f / 2.2f));

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0f), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0f), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0f), 0, 255);

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
static int* dev_material_keys = NULL;
static int* dev_light_indices = NULL;
static int light_count = 0;
static Triangle* dev_triangles = NULL;
static MeshBVHNode* dev_mesh_nodes = NULL;
static bool statisticsEnabled = false;
static std::vector<BounceStatistics> bounceStatistics;
static cudaEvent_t stageStart = NULL, stageStop = NULL;
static cudaEvent_t traceStart = NULL;
static cudaEvent_t traceStop = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void enableTraceStatistics(bool enabled) { statisticsEnabled = enabled; }
const std::vector<BounceStatistics>& lastBounceStatistics() { return bounceStatistics; }

static void beginStage()
{
    if (statisticsEnabled) cudaEventRecord(stageStart);
}

static float endStage()
{
    if (!statisticsEnabled) return 0.0f;
    cudaEventRecord(stageStop);
    cudaEventSynchronize(stageStop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, stageStart, stageStop);
    return elapsed;
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;
    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_mesh_nodes, scene->meshNodes.size() * sizeof(MeshBVHNode));
        cudaMemcpy(dev_mesh_nodes, scene->meshNodes.data(), scene->meshNodes.size() * sizeof(MeshBVHNode), cudaMemcpyHostToDevice);
    }

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

    std::vector<int> lightIndices;
    for (int i = 0; i < static_cast<int>(scene->geoms.size()); ++i)
    {
        const Geom& geom = scene->geoms[i];
        if (scene->materials[geom.materialid].emittance > 0.0f &&
            (geom.type == CUBE || geom.type == SPHERE))
        {
            lightIndices.push_back(i);
        }
    }
    light_count = static_cast<int>(lightIndices.size());
    if (light_count > 0)
    {
        cudaMalloc(&dev_light_indices, light_count * sizeof(int));
        cudaMemcpy(dev_light_indices, lightIndices.data(),
            light_count * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaEventCreate(&traceStart);
    cudaEventCreate(&traceStop);
    cudaEventCreate(&stageStart);
    cudaEventCreate(&stageStop);

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
    cudaFree(dev_light_indices);
    cudaFree(dev_triangles);
    cudaFree(dev_mesh_nodes);
    dev_triangles = NULL;
    dev_mesh_nodes = NULL;
    dev_light_indices = NULL;
    light_count = 0;
    if (traceStart != NULL) cudaEventDestroy(traceStart);
    if (traceStop != NULL) cudaEventDestroy(traceStop);
    traceStart = NULL;
    traceStop = NULL;
    if (stageStart != NULL) cudaEventDestroy(stageStart);
    if (stageStop != NULL) cudaEventDestroy(stageStop);
    stageStart = NULL;
    stageStop = NULL;

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antialiasing)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;
        if (!antialiasing) jitterX = jitterY = 0.0f;
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + 0.5f + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + 0.5f + jitterY - (float)cam.resolution.y * 0.5f)
        );

        if (cam.aperture > 0.0f)
        {
            float radius = cam.aperture * sqrtf(u01(rng));
            float angle = TWO_PI * u01(rng);
            glm::vec3 focusPoint = cam.position + segment.ray.direction *
                (cam.focusDistance / glm::max(glm::dot(segment.ray.direction, cam.view), 0.0001f));
            segment.ray.origin += radius *
                (cosf(angle) * cam.right + sinf(angle) * cam.up);
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.specularBounce = true;
    }
}

__global__ void generateMaterialKeys(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    Material* materials,
    int* materialKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        int key = -1;
        if (intersection.t > 0.0f)
        {
            Material material = materials[intersection.materialId];
            key = material.emittance > 0.0f ? 0 :
                material.hasRefractive > 0.0f ? 1 :
                material.hasReflective > 0.0f ? 2 : 3;
        }
        materialKeys[idx] = key;
    }
}

struct PathSegmentTerminated
{
    __host__ __device__ bool operator()(const PathSegment& pathSegment) const
    {
        return pathSegment.remainingBounces == 0;
    }
};

__device__ float intersectGeometry(
    const Geom& geom,
    const Ray& ray,
    glm::vec3& point,
    glm::vec3& normal,
    bool& outside,
    const Triangle* triangles,
    const MeshBVHNode* nodes,
    bool useBVH,
    float maxDistance = FLT_MAX)
{
    if (geom.type == MESH)
    {
        return meshIntersectionTest(geom, ray, triangles, nodes, useBVH,
            point, normal, outside, maxDistance);
    }
    if (geom.type == CUBE)
    {
        return boxIntersectionTest(geom, ray, point, normal, outside);
    }
    if (geom.type == SPHERE)
    {
        return sphereIntersectionTest(geom, ray, point, normal, outside);
    }
    return proceduralIntersectionTest(geom, ray, point, normal, outside);
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
    const Triangle* triangles,
    const MeshBVHNode* nodes,
    bool useBVH)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        if (pathSegment.remainingBounces == 0)
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
        bool hitOutside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            t = intersectGeometry(geom, pathSegment.ray,
                tmp_intersect, tmp_normal, outside, triangles, nodes, useBVH, t_min);
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
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomIndex = hit_geom_index;
            intersections[path_index].objectPoint = multiplyMV(
                geoms[hit_geom_index].inverseTransform,
                glm::vec4(intersect_point, 1.0f));
            intersections[path_index].outside = hitOutside;
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

__device__ glm::vec3 materialColorAt(const Material& material, glm::vec3 p)
{
    if (material.texture == CHECKER)
    {
        glm::ivec3 cell(
            static_cast<int>(floorf(p.x * material.textureScale)),
            static_cast<int>(floorf(p.y * material.textureScale)),
            static_cast<int>(floorf(p.z * material.textureScale)));
        return ((cell.x + cell.y + cell.z) & 1) ?
            material.color : material.secondaryColor;
    }
    if (material.texture == MARBLE)
    {
        float bands = sinf(material.textureScale * p.y
            + 2.0f * sinf(material.textureScale * p.x * 0.55f)
            + 1.2f * sinf(material.textureScale * p.z * 0.8f));
        float blend = 0.5f + 0.5f * bands;
        return glm::mix(material.color, material.secondaryColor, blend);
    }
    return material.color;
}

__device__ void sampleLightSurface(
    const Geom& light,
    thrust::default_random_engine& rng,
    glm::vec3& position,
    glm::vec3& normal,
    float& pdfArea)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    glm::vec3 localPoint;
    glm::vec3 localNormal;
    if (light.type == CUBE)
    {
        glm::vec3 axisX = multiplyMV(light.transform, glm::vec4(1, 0, 0, 0));
        glm::vec3 axisY = multiplyMV(light.transform, glm::vec4(0, 1, 0, 0));
        glm::vec3 axisZ = multiplyMV(light.transform, glm::vec4(0, 0, 1, 0));
        float areaYZ = glm::length(glm::cross(axisY, axisZ));
        float areaXZ = glm::length(glm::cross(axisX, axisZ));
        float areaXY = glm::length(glm::cross(axisX, axisY));
        float totalArea = 2.0f * (areaYZ + areaXZ + areaXY);
        float face = u01(rng) * totalArea;
        float u = u01(rng) - 0.5f;
        float v = u01(rng) - 0.5f;
        if (face < areaYZ)
        {
            localPoint = glm::vec3(0.5f, u, v);
            localNormal = glm::vec3(1, 0, 0);
        }
        else if (face < 2.0f * areaYZ)
        {
            localPoint = glm::vec3(-0.5f, u, v);
            localNormal = glm::vec3(-1, 0, 0);
        }
        else if (face < 2.0f * areaYZ + areaXZ)
        {
            localPoint = glm::vec3(u, 0.5f, v);
            localNormal = glm::vec3(0, 1, 0);
        }
        else if (face < 2.0f * (areaYZ + areaXZ))
        {
            localPoint = glm::vec3(u, -0.5f, v);
            localNormal = glm::vec3(0, -1, 0);
        }
        else if (face < 2.0f * (areaYZ + areaXZ) + areaXY)
        {
            localPoint = glm::vec3(u, v, 0.5f);
            localNormal = glm::vec3(0, 0, 1);
        }
        else
        {
            localPoint = glm::vec3(u, v, -0.5f);
            localNormal = glm::vec3(0, 0, -1);
        }
        pdfArea = 1.0f / totalArea;
    }
    else
    {
        float z = 1.0f - 2.0f * u01(rng);
        float angle = TWO_PI * u01(rng);
        float radius = sqrtf(glm::max(0.0f, 1.0f - z * z));
        localNormal = glm::vec3(radius * cosf(angle), z, radius * sinf(angle));
        localPoint = 0.5f * localNormal;
        float jacobian = fabsf(light.scale.x * light.scale.y * light.scale.z) *
            glm::length(multiplyMV(light.invTranspose, glm::vec4(localNormal, 0.0f)));
        pdfArea = 1.0f / (PI * jacobian);
    }
    position = multiplyMV(light.transform, glm::vec4(localPoint, 1.0f));
    normal = glm::normalize(multiplyMV(light.invTranspose,
        glm::vec4(localNormal, 0.0f)));
}

__device__ glm::vec3 estimateDirectLighting(
    glm::vec3 point,
    glm::vec3 normal,
    glm::vec3 albedo,
    Geom* geoms,
    int geomCount,
    Material* materials,
    int* lightIndices,
    int lightCount,
    thrust::default_random_engine& rng,
    const Triangle* triangles,
    const MeshBVHNode* nodes,
    bool useBVH)
{
    if (lightCount == 0) return glm::vec3(0.0f);

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    int lightIndex = glm::min(static_cast<int>(u01(rng) * lightCount), lightCount - 1);
    Geom light = geoms[lightIndices[lightIndex]];
    glm::vec3 lightPosition;
    glm::vec3 lightNormal;
    float pdfArea;
    sampleLightSurface(light, rng, lightPosition, lightNormal, pdfArea);

    glm::vec3 toLight = lightPosition - point;
    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= 1e-8f) return glm::vec3(0.0f);
    float distance = sqrtf(distanceSquared);
    glm::vec3 direction = toLight / distance;
    float cosSurface = glm::max(0.0f, glm::dot(normal, direction));
    float cosLight = glm::max(0.0f, glm::dot(lightNormal, -direction));
    if (cosSurface == 0.0f || cosLight == 0.0f) return glm::vec3(0.0f);

    Ray shadowRay;
    shadowRay.origin = point + normal * 0.004f;
    shadowRay.direction = direction;
    for (int i = 0; i < geomCount; ++i)
    {
        glm::vec3 hitPoint;
        glm::vec3 hitNormal;
        bool outside;
        float hitDistance = intersectGeometry(geoms[i], shadowRay,
            hitPoint, hitNormal, outside, triangles, nodes, useBVH, distance - 0.005f);
        if (hitDistance > 0.0001f && hitDistance < distance - 0.005f)
        {
            return glm::vec3(0.0f);
        }
    }

    Material emitter = materials[light.materialid];
    return albedo * emitter.color * emitter.emittance *
        (cosSurface * cosLight * lightCount /
            (PI * distanceSquared * pdfArea));
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geomCount,
    int* lightIndices,
    int lightCount,
    bool directLightingEnabled,
    bool russianRouletteEnabled,
    glm::vec3* image,
    const Triangle* triangles,
    const MeshBVHNode* nodes,
    bool useBVH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];
        if (pathSegment.remainingBounces == 0) return;

        if (intersection.t <= 0.0f)
        {
            pathSegment.color = BACKGROUND_COLOR;
            pathSegment.remainingBounces = 0;
            return;
        }

        Material material = materials[intersection.materialId];
        if (material.emittance > 0.0f)
        {
            if (directLightingEnabled && lightCount > 0 && !pathSegment.specularBounce &&
                (geoms[intersection.geomIndex].type == CUBE ||
                    geoms[intersection.geomIndex].type == SPHERE))
            {
                pathSegment.color = glm::vec3(0.0f);
            }
            else
            {
                pathSegment.color *= material.color * material.emittance;
            }
            pathSegment.remainingBounces = 0;
            return;
        }

        material.color = materialColorAt(material, intersection.objectPoint);
        glm::vec3 intersectionPoint = pathSegment.ray.origin +
            intersection.t * pathSegment.ray.direction;
        thrust::default_random_engine rng = makeSeededRandomEngine(
            iter, pathSegment.pixelIndex, depth);

        if (directLightingEnabled && material.hasReflective == 0.0f &&
            material.hasRefractive == 0.0f)
        {
            image[pathSegment.pixelIndex] += pathSegment.color *
                estimateDirectLighting(
                    intersectionPoint, intersection.surfaceNormal,
                    material.color, geoms, geomCount, materials,
                    lightIndices, lightCount, rng, triangles, nodes, useBVH);
        }

        if (pathSegment.remainingBounces <= 1)
        {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        scatterRay(
            pathSegment,
            intersectionPoint,
            intersection.surfaceNormal,
            intersection.outside,
            material,
            rng);
        pathSegment.remainingBounces--;

        if (russianRouletteEnabled && depth >= 3 &&
            pathSegment.remainingBounces > 0)
        {
            float survival = glm::clamp(
                glm::max(pathSegment.color.x,
                    glm::max(pathSegment.color.y, pathSegment.color.z)),
                0.05f, 0.95f);
            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
            if (u01(rng) > survival)
            {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
            }
            else
            {
                pathSegment.color /= survival;
            }
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
        if (iterationPath.remainingBounces == 0)
        {
            image[iterationPath.pixelIndex] += iterationPath.color;
            // Uncompacted dead paths must contribute only once, too.
            iterationPaths[index].color = glm::vec3(0.0f);
        }
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

    bounceStatistics.clear();
    const bool useBVH = guiData == NULL || guiData->MeshBVHEnabled;
    const bool compact = guiData == NULL || guiData->CompactionEnabled;
    const int seededIteration = iter + (guiData == NULL ? 0 : guiData->SeedOffset);
    cudaEventRecord(traceStart);
    beginStage();
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, seededIteration, traceDepth,
        dev_paths, guiData == NULL || guiData->AntialiasingEnabled);
    checkCUDAError("generate camera ray");
    float cameraMs = endStage();

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        BounceStatistics stats;
        stats.depth = depth + 1;
        stats.raysScheduled = num_paths;
        stats.raysIn = depth == 0 ? pixelcount :
            (statisticsEnabled ? bounceStatistics.back().raysAlive : num_paths);
        stats.cameraMs = depth == 0 ? cameraMs : 0.0f;
        beginStage();
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
            dev_intersections,
            dev_triangles,
            dev_mesh_nodes,
            useBVH
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        stats.intersectionMs = endStage();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        if (guiData == NULL || guiData->MaterialSortingEnabled)
        {
            beginStage();
            generateMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths,
                dev_intersections,
                dev_materials,
                dev_material_keys
            );
            checkCUDAError("generate material sort keys");

            auto zippedPathsAndIntersections = thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::device_pointer_cast(dev_paths),
                    thrust::device_pointer_cast(dev_intersections)));
            thrust::sort_by_key(
                thrust::device,
                thrust::device_pointer_cast(dev_material_keys),
                thrust::device_pointer_cast(dev_material_keys) + num_paths,
                zippedPathsAndIntersections);
            checkCUDAError("sort paths by material");
            stats.sortingMs = endStage();
        }

        beginStage();
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            seededIteration,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            dev_light_indices,
            light_count,
            guiData == NULL || guiData->DirectLightingEnabled,
            guiData == NULL || guiData->RussianRouletteEnabled,
            dev_image,
            dev_triangles,
            dev_mesh_nodes,
            useBVH
        );
        checkCUDAError("shade one bounce");
        stats.shadingMs = endStage();

        beginStage();
        finalGather<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, dev_image, dev_paths);
        checkCUDAError("gather terminated paths");
        stats.gatherMs = endStage();

        if (statisticsEnabled)
        {
            stats.raysAlive = num_paths - static_cast<int>(thrust::count_if(
                thrust::device, dev_paths, dev_paths + num_paths, PathSegmentTerminated()));
        }
        if (compact)
        {
            beginStage();
            PathSegment* dev_path_end = thrust::remove_if(
                thrust::device,
                dev_paths,
                dev_paths + num_paths,
                PathSegmentTerminated());
            num_paths = static_cast<int>(dev_path_end - dev_paths);
            stats.compactionMs = endStage();
        }
        iterationComplete = compact ? num_paths == 0 : depth >= traceDepth; // TODO: should be based off stream compaction results.
        if (statisticsEnabled) bounceStatistics.push_back(stats);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    if (pbo != NULL)
    {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    }
    cudaEventRecord(traceStop);
    cudaEventSynchronize(traceStop);
    if (guiData != NULL)
    {
        cudaEventElapsedTime(&guiData->LastTraceMs, traceStart, traceStop);
    }

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
