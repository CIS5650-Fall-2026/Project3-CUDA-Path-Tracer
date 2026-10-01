#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thrust/sort.h>

#define ERRORCHECK 1
#define RUSSIAN_ROULETTE_START_DEPTH 3

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

// Halton dims: pixel (2), lens (2), time (1), then 2 per bounce
#define HALTON_DIMENSIONS 13
#define HALTON_CAMERA_DIMENSIONS 5
#define HALTON_BOUNCES ((HALTON_DIMENSIONS - HALTON_CAMERA_DIMENSIONS) / 2)

__constant__ int haltonPrimes[HALTON_DIMENSIONS] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41 };

__device__ float radicalInverse(int base, unsigned int index)
{
    const float invBase = 1.0f / base;
    float digitWeight = invBase;
    float result = 0.0f;

    while (index > 0)
    {
        result += digitWeight * (index % base);
        index /= base;
        digitWeight *= invBase;
    }

    return result;
}

// Shift the points per pixel so pixels aren't correlated
__device__ float haltonSample(int dimension, int iter, int pixelIndex)
{
    const unsigned int hash = utilhash(pixelIndex * HALTON_DIMENSIONS + dimension);
    const float shift = hash * (1.0f / 4294967296.0f);
    const float value = radicalInverse(haltonPrimes[dimension], iter) + shift;
    return value - floorf(value);
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
static int* dev_lights = NULL;
static int numLights = 0;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhNodes = NULL;

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

    const int numTriangles = static_cast<int>(scene->triangles.size());
    cudaMalloc(&dev_triangles, std::max(numTriangles, 1) * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

    const int numNodes = static_cast<int>(scene->bvhNodes.size());
    cudaMalloc(&dev_bvhNodes, std::max(numNodes, 1) * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), numNodes * sizeof(BVHNode), cudaMemcpyHostToDevice);

    // Only cubes and spheres can be sampled as lights
    std::vector<int> lights;
    for (int i = 0; i < static_cast<int>(scene->geoms.size()); ++i)
    {
        const GeomType type = scene->geoms[i].type;
        if (scene->materials[scene->geoms[i].materialid].emittance > 0.0f &&
            (type == CUBE || type == SPHERE))
        {
            lights.push_back(i);
        }
    }
    numLights = static_cast<int>(lights.size());

    cudaMalloc(&dev_lights, std::max(numLights, 1) * sizeof(int));
    cudaMemcpy(dev_lights, lights.data(), numLights * sizeof(int), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_lights);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);

    checkCUDAError("pathtraceFree");
}

// Used when resuming from a checkpoint
void pathtraceSetImage(const glm::vec3* image)
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMemcpy(dev_image, image, pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("pathtraceSetImage");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
    Camera cam,
    int iter,
    int traceDepth,
    PathSegment* pathSegments,
    bool useMotionBlur,
    bool useHalton,
    bool useAntialiasing)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y)
    {
        return;
    }

    int index = x + y * cam.resolution.x;
    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f);
    segment.radiance = glm::vec3(0.0f);
    segment.bsdfPdf = 0.0f;

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float u[HALTON_CAMERA_DIMENSIONS];
    for (int i = 0; i < HALTON_CAMERA_DIMENSIONS; ++i)
    {
        u[i] = useHalton ? haltonSample(i, iter, index) : u01(rng);
    }

    if (!useAntialiasing)
    {
        u[0] = 0.5f;
        u[1] = 0.5f;
    }

    float sampleX = static_cast<float>(x) + u[0];
    float sampleY = static_cast<float>(y) + u[1];

    segment.ray.direction = glm::normalize(
        cam.view
        - cam.right * cam.pixelLength.x
            * (sampleX - cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y
            * (sampleY - cam.resolution.y * 0.5f));
    
    if (cam.apertureRadius > 0.0f)
    {
        const glm::vec3 forward = glm::normalize(cam.view);

        // Find where the pinhole ray reaches the focal plane.
        const float focusT =
            cam.focalDistance /
            glm::dot(segment.ray.direction, forward);

        const glm::vec3 focusPoint =
            cam.position + focusT * segment.ray.direction;

        // Uniformly sample the area of a circular aperture.
        const float radius =
            cam.apertureRadius * sqrtf(u[2]);
        const float angle = TWO_PI * u[3];

        const glm::vec3 lensOffset =
            cam.right * (radius * cosf(angle)) +
            cam.up * (radius * sinf(angle));

        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction =
            glm::normalize(focusPoint - segment.ray.origin);
    }

    segment.time = useMotionBlur ? u[4] : 0.0f;

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
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
    MeshData meshData,
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

            // Motion blur: move the ray back instead of the object
            Ray ray = pathSegment.ray;
            ray.origin -= geom.velocity * pathSegment.time;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, meshData, ray, tmp_intersect, tmp_normal, outside);
            }
            else
            {
                t = fractalIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
            }

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
            intersections[path_index].geomId = hit_geom_index;
        }
    }
}

__device__ float valueNoise(glm::vec3 p)
{
    const glm::vec3 cell = glm::floor(p);
    const glm::vec3 f = p - cell;
    const glm::vec3 w = f * f * (3.0f - 2.0f * f);

    float corners[8];
    for (int i = 0; i < 8; ++i)
    {
        const glm::ivec3 c = glm::ivec3(cell) + glm::ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        corners[i] = utilhash(c.x * 73856093 ^ c.y * 19349663 ^ c.z * 83492791) * (1.0f / 4294967296.0f);
    }

    const float x00 = glm::mix(corners[0], corners[1], w.x);
    const float x10 = glm::mix(corners[2], corners[3], w.x);
    const float x01 = glm::mix(corners[4], corners[5], w.x);
    const float x11 = glm::mix(corners[6], corners[7], w.x);
    return glm::mix(glm::mix(x00, x10, w.y), glm::mix(x01, x11, w.y), w.z);
}

__device__ float fbm(glm::vec3 p)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    for (int octave = 0; octave < 5; ++octave)
    {
        sum += amplitude * valueNoise(p);
        p *= 2.0f;
        amplitude *= 0.5f;
    }
    return sum;
}

__device__ glm::vec3 textureColor(const Material& m, glm::vec3 p)
{
    const glm::vec3 q = p * m.textureScale;

    if (m.texture == TEXTURE_CHECKER)
    {
        const glm::ivec3 cell = glm::ivec3(glm::floor(q));
        return ((cell.x + cell.y + cell.z) & 1) ? m.color : m.color2;
    }

    // Marble: stripes bent by noise
    const float stripes = 0.5f + 0.5f * sinf(q.x * 2.0f + 8.0f * fbm(q));
    return glm::mix(m.color2, m.color, stripes);
}

__device__ float geomSurfaceArea(const Geom& geom)
{
    const glm::vec3 s = geom.scale;

    if (geom.type == CUBE)
    {
        return 2.0f * (s.y * s.z + s.x * s.z + s.x * s.y);
    }

    const float radius = 0.5f * s.x;
    return 4.0f * PI * radius * radius;
}

__device__ float powerHeuristic(float pdfA, float pdfB)
{
    return pdfA * pdfA / (pdfA * pdfA + pdfB * pdfB);
}

__device__ float lightPdf(const Geom& light, float dist2, float cosLight, int numLights)
{
    return dist2 / (cosLight * geomSurfaceArea(light) * numLights);
}

// Uniformly samples a point on the surface of a cube or sphere.
__device__ glm::vec3 sampleGeomSurface(
    const Geom& geom,
    thrust::default_random_engine& rng,
    glm::vec3& normal)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    glm::vec3 localPoint;
    glm::vec3 localNormal;

    if (geom.type == CUBE)
    {
        const glm::vec3 s = geom.scale;
        const float areaX = s.y * s.z;
        const float areaY = s.x * s.z;
        const float areaZ = s.x * s.y;

        const float pick = u01(rng) * (areaX + areaY + areaZ);
        const float side = u01(rng) < 0.5f ? -0.5f : 0.5f;
        const float a = u01(rng) - 0.5f;
        const float b = u01(rng) - 0.5f;

        if (pick < areaX)
        {
            localPoint = glm::vec3(side, a, b);
            localNormal = glm::vec3(side, 0.0f, 0.0f);
        }
        else if (pick < areaX + areaY)
        {
            localPoint = glm::vec3(a, side, b);
            localNormal = glm::vec3(0.0f, side, 0.0f);
        }
        else
        {
            localPoint = glm::vec3(a, b, side);
            localNormal = glm::vec3(0.0f, 0.0f, side);
        }
    }
    else
    {
        const float z = 1.0f - 2.0f * u01(rng);
        const float r = sqrtf(glm::max(0.0f, 1.0f - z * z));
        const float phi = TWO_PI * u01(rng);

        localNormal = glm::vec3(r * cosf(phi), r * sinf(phi), z);
        localPoint = 0.5f * localNormal;
    }

    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.0f)));
    return multiplyMV(geom.transform, glm::vec4(localPoint, 1.0f));
}

__device__ bool isOccluded(
    const Ray& shadowRay,
    float time,
    float maxT,
    Geom* geoms,
    int geoms_size,
    MeshData meshData,
    int ignoreGeom)
{
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    bool outside;

    for (int i = 0; i < geoms_size; i++)
    {
        if (i == ignoreGeom)
        {
            continue;
        }

        float t = -1.0f;

        Ray ray = shadowRay;
        ray.origin -= geoms[i].velocity * time;

        if (geoms[i].type == CUBE)
        {
            t = boxIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geoms[i].type == SPHERE)
        {
            t = sphereIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geoms[i].type == MESH)
        {
            t = meshIntersectionTest(geoms[i], meshData, ray, tmp_intersect, tmp_normal, outside);
        }
        else
        {
            t = fractalIntersectionTest(geoms[i], ray, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f && t < maxT)
        {
            return true;
        }
    }

    return false;
}

// Direct lighting with MIS
__device__ glm::vec3 sampleDirectLight(
    glm::vec3 hitPoint,
    glm::vec3 normal,
    float time,
    float fogDensity,
    const Material& material,
    Geom* geoms,
    int geoms_size,
    MeshData meshData,
    Material* materials,
    const int* lights,
    int numLights,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    const int lightIndex =
        lights[glm::min(static_cast<int>(u01(rng) * numLights), numLights - 1)];
    const Geom& light = geoms[lightIndex];

    glm::vec3 lightNormal;
    const glm::vec3 lightPoint =
        sampleGeomSurface(light, rng, lightNormal) + light.velocity * time;

    const glm::vec3 toLight = lightPoint - hitPoint;
    const float dist2 = glm::dot(toLight, toLight);
    const float dist = sqrtf(dist2);
    const glm::vec3 wi = toLight / dist;

    const float cosSurface = glm::dot(normal, wi);
    const float cosLight = -glm::dot(lightNormal, wi);

    if (cosSurface <= 0.0f || cosLight <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    Ray shadowRay;
    shadowRay.origin = hitPoint + 0.001f * normal;
    shadowRay.direction = wi;

    if (isOccluded(shadowRay, time, dist - 0.002f, geoms, geoms_size, meshData, lightIndex))
    {
        return glm::vec3(0.0f);
    }

    const Material& lightMaterial = materials[light.materialid];
    const glm::vec3 emitted = lightMaterial.color * lightMaterial.emittance;

    const float pdfLight = lightPdf(light, dist2, cosLight, numLights);
    const float pdfBsdf = cosSurface / PI;

    const float transmittance = expf(-fogDensity * dist);

    return material.color / PI * emitted * cosSurface / pdfLight
        * powerHeuristic(pdfLight, pdfBsdf) * transmittance;
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
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    MeshData meshData,
    const int* lights,
    int numLights,
    bool useDirectLighting,
    bool useRussianRoulette,
    bool useHalton,
    Fog fog)
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

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, path.pixelIndex, depth + 1);

    // Fog: scatter here if the sampled distance is before the hit
    if (fog.density > 0.0f && intersection.t > 0.0f)
    {
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float scatterT = -logf(1.0f - u01(rng)) / fog.density;

        if (scatterT < intersection.t)
        {
            const float z = 1.0f - 2.0f * u01(rng);
            const float r = sqrtf(glm::max(0.0f, 1.0f - z * z));
            const float phi = TWO_PI * u01(rng);

            path.ray.origin += scatterT * path.ray.direction;
            path.ray.direction = glm::vec3(r * cosf(phi), r * sinf(phi), z);
            path.color *= fog.albedo;
            path.bsdfPdf = 0.0f;
            --path.remainingBounces;
            return;
        }
    }

    if (intersection.t <= 0.0f)
    {
        path.remainingBounces = 0;
        return;
    }

    const Material& material = materials[intersection.materialId];

    if (material.emittance > 0.0f)
    {
        float weight = 1.0f;
        const float cosLight =
            -glm::dot(intersection.surfaceNormal, path.ray.direction);

        const GeomType lightType = geoms[intersection.geomId].type;
        if (useDirectLighting && path.bsdfPdf > 0.0f && cosLight > 0.0f &&
            (lightType == CUBE || lightType == SPHERE))
        {
            const float dist2 = intersection.t * intersection.t;
            const float pdfLight = lightPdf(
                geoms[intersection.geomId], dist2, cosLight, numLights);
            weight = powerHeuristic(path.bsdfPdf, pdfLight);
        }

        path.radiance +=
            weight * path.color * material.color * material.emittance;
        path.remainingBounces = 0;
        return;
    }

    glm::vec3 hitPoint =
        path.ray.origin + intersection.t * path.ray.direction;

    const bool isDiffuse =
        material.hasReflective <= 0.0f && material.hasRefractive <= 0.0f;

    Material shadingMaterial = material;
    if (material.texture != TEXTURE_NONE)
    {
        const glm::vec3 texturePoint =
            hitPoint - geoms[intersection.geomId].velocity * path.time;
        shadingMaterial.color = textureColor(material, texturePoint);
    }

    glm::vec3 normal = intersection.surfaceNormal;
    if (glm::dot(normal, path.ray.direction) > 0.0f)
    {
        normal = -normal;
    }

    // Not on the last bounce, it would go past the max depth
    if (useDirectLighting && isDiffuse && numLights > 0 &&
        path.remainingBounces > 1)
    {
        path.radiance += path.color * sampleDirectLight(
            hitPoint, normal, path.time, fog.density, shadingMaterial, geoms, geoms_size,
            meshData, materials, lights, numLights, rng);
    }

    glm::vec2 haltonDirection;
    const bool haltonBounce = useHalton && depth < HALTON_BOUNCES;
    if (haltonBounce)
    {
        const int dimension = HALTON_CAMERA_DIMENSIONS + 2 * depth;
        haltonDirection = glm::vec2(
            haltonSample(dimension, iter, path.pixelIndex),
            haltonSample(dimension + 1, iter, path.pixelIndex));
    }

    scatterRay(
        path,
        hitPoint,
        intersection.surfaceNormal,
        shadingMaterial,
        rng,
        haltonBounce ? &haltonDirection : nullptr);

    path.bsdfPdf = isDiffuse
        ? glm::max(glm::dot(normal, path.ray.direction), 0.0f) / PI
        : 0.0f;

    if (useRussianRoulette &&
        depth + 1 >= RUSSIAN_ROULETTE_START_DEPTH &&
        path.remainingBounces > 0)
    {
        const float survival = glm::min(
            glm::max(path.color.r, glm::max(path.color.g, path.color.b)),
            1.0f);

        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        if (u01(rng) >= survival)
        {
            path.remainingBounces = 0;
        }
        else
        {
            path.color /= survival;
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

__global__ void gatherTerminatedPaths(
    int numPaths,
    glm::vec3* image,
    const PathSegment* paths)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numPaths)
    {
        return;
    }

    const PathSegment& path = paths[index];

    if (path.remainingBounces <= 0)
    {
        image[path.pixelIndex] += path.radiance;
    }
}

struct IsTerminated
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces <= 0;
    }
};

struct CompareMaterial
{
    __host__ __device__ bool operator()(
        const ShadeableIntersection& a,
        const ShadeableIntersection& b) const
    {
        int materialA = a.t > 0.0f ? a.materialId : -1;
        int materialB = b.t > 0.0f ? b.materialId : -1;

        return materialA < materialB;
    }
};

static bool envFlag(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
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

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    const bool sortMaterials = envFlag("PROJECT3_SORT_MATERIALS");
    const bool useRussianRoulette = envFlag("PROJECT3_RUSSIAN_ROULETTE");
    const bool useCompaction = !envFlag("PROJECT3_NO_COMPACTION");
    const bool useDirectLighting = envFlag("PROJECT3_DIRECT_LIGHTING");
    const bool useMotionBlur = envFlag("PROJECT3_MOTION_BLUR");
    const bool useHalton = envFlag("PROJECT3_HALTON");
    const bool useAntialiasing = !envFlag("PROJECT3_NO_ANTIALIAS");

    MeshData meshData;
    meshData.triangles = dev_triangles;
    meshData.nodes = dev_bvhNodes;
    meshData.useBoundsCulling = envFlag("PROJECT3_MESH_CULLING");
    meshData.useBVH = envFlag("PROJECT3_BVH");

    Fog fog = hst_scene->fog;
    if (!envFlag("PROJECT3_FOG"))
    {
        fog.density = 0.0f;
    }
    const bool printPathCounts =
        iter == 1 && envFlag("PROJECT3_PRINT_PATH_COUNTS");

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, dev_paths, useMotionBlur, useHalton, useAntialiasing);
    checkCUDAError("generate camera ray");

    int numPaths = pixelcount;

    if (printPathCounts)
    {
        printf("Paths after bounce 0: %d\n", numPaths);
    }

    for (int depth = 0;
         depth < traceDepth && numPaths > 0;
         ++depth)
    {
        int numBlocks =
            (numPaths + blockSize1d - 1) / blockSize1d;

        computeIntersections<<<numBlocks, blockSize1d>>>(
            depth,
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            meshData,
            dev_intersections);
        checkCUDAError("compute intersections");

        if (sortMaterials)
        {
            thrust::sort_by_key(
                thrust::device,
                dev_intersections,
                dev_intersections + numPaths,
                dev_paths,
                CompareMaterial());

            checkCUDAError("sort paths by material");
        }

        shadeMaterial<<<numBlocks, blockSize1d>>>(
            iter,
            depth,
            numPaths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            meshData,
            dev_lights,
            numLights,
            useDirectLighting,
            useRussianRoulette,
            useHalton,
            fog);
        checkCUDAError("shade material");

        // Without compaction, gather everything once after the loop
        if (useCompaction)
        {
            gatherTerminatedPaths<<<numBlocks, blockSize1d>>>(
                numPaths,
                dev_image,
                dev_paths);
            checkCUDAError("gather terminated paths");

            PathSegment* newEnd = thrust::remove_if(
                thrust::device,
                dev_paths,
                dev_paths + numPaths,
                IsTerminated());

            numPaths = static_cast<int>(newEnd - dev_paths);
            checkCUDAError("compact paths");
        }

        if (printPathCounts)
        {
            printf("Paths after bounce %d: %d\n", depth + 1, numPaths);
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth + 1;
        }
    }

    if (!useCompaction)
    {
        gatherTerminatedPaths<<<(pixelcount + blockSize1d - 1) / blockSize1d, blockSize1d>>>(
            pixelcount,
            dev_image,
            dev_paths);
        checkCUDAError("gather terminated paths");
    }

    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo, cam.resolution, iter, dev_image);

    cudaMemcpy(
        hst_scene->state.image.data(),
        dev_image,
        pixelcount * sizeof(glm::vec3),
        cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}