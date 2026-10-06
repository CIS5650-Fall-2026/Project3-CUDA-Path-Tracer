#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

__device__ int d_badRays;   // diagnostic: rays with non-finite origin/direction

#define ENV_NEE 1
#define LIGHT_NEE 1

#define ERRORCHECK 1
#define RUSSIAN_ROULETTE 1

#define STREAM_COMPACTION 1
#define PROFILE 1             // time each stage with CUDA events
#define LOG_BOUNCE_ITER 10    // print alive paths per bounce on this iteration
#define PROFILE_WINDOW 200    // print average stage times every N iterations

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
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool toneMap,
    const glm::vec3* bloom, float bloomStrength)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        glm::vec3 hdr = pix / (float)iter;
        if (bloom) hdr += bloomStrength * bloom[index];
        glm::vec3 d = toDisplay(hdr, toneMap);
        color.x = (int)(d.x * 255.0f);
        color.y = (int)(d.y * 255.0f);
        color.z = (int)(d.z * 255.0f);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// Keep only the energy above the threshold (linear HDR, before tone mapping)
__global__ void brightPass(const glm::vec3* image, glm::vec3* out, int n, int iter, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    glm::vec3 c = image[i] / (float)iter;
    out[i] = glm::max(c - glm::vec3(threshold), glm::vec3(0.0f));
}

// One direction of a separable Gaussian blur (run twice: horizontal, then vertical)
__global__ void blurPass(const glm::vec3* in, glm::vec3* out, int w, int h,
    int radius, float sigma, bool horizontal)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    glm::vec3 sum(0.0f);
    float wsum = 0.0f;
    for (int k = -radius; k <= radius; k++)
    {
        int xx = horizontal ? min(max(x + k, 0), w - 1) : x;
        int yy = horizontal ? y : min(max(y + k, 0), h - 1);
        float wgt = expf(-(float)(k * k) / (2.0f * sigma * sigma));
        sum += wgt * in[yy * w + xx];
        wsum += wgt;
    }
    out[y * w + x] = sum / wsum;
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhNodes = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static glm::vec3* dev_envMap = NULL;
static float* dev_envCdf = NULL;
static glm::vec3* dev_texPixels = NULL;
static TextureInfo* dev_texInfo = NULL;
static int* dev_lights = NULL;
static glm::vec3* dev_bloomA = NULL;
static glm::vec3* dev_bloomB = NULL;

#if PROFILE
static cudaEvent_t evStart = NULL, evStop = NULL;
static double timeIntersect = 0.0, timeSort = 0.0, timeShade = 0.0, timeCompact = 0.0;
static int profiledIters = 0;

// Milliseconds since the last PROFILE_BEGIN()
static float elapsedMs()
{
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, evStart, evStop);
    return ms;
}
#define PROFILE_BEGIN() cudaEventRecord(evStart)
#define PROFILE_END(acc) (acc) += elapsedMs()
#else
#define PROFILE_BEGIN()
#define PROFILE_END(acc)
#endif

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

    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(),
            scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    if (!scene->bvhNodes.empty())
    {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(),
            scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }
    if (!scene->envMap.empty())
    {
        cudaMalloc(&dev_envMap, scene->envMap.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_envMap, scene->envMap.data(),
            scene->envMap.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    if (!scene->envCdf.empty())
    {
        cudaMalloc(&dev_envCdf, scene->envCdf.size() * sizeof(float));
        cudaMemcpy(dev_envCdf, scene->envCdf.data(),
            scene->envCdf.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (!scene->textures.empty())
    {
        cudaMalloc(&dev_texPixels, scene->texPixels.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_texPixels, scene->texPixels.data(),
            scene->texPixels.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_texInfo, scene->textures.size() * sizeof(TextureInfo));
        cudaMemcpy(dev_texInfo, scene->textures.data(),
            scene->textures.size() * sizeof(TextureInfo), cudaMemcpyHostToDevice);
    }

    if (!scene->lights.empty())
    {
        cudaMalloc(&dev_lights, scene->lights.size() * sizeof(int));
        cudaMemcpy(dev_lights, scene->lights.data(),
            scene->lights.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_bloomA, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_bloomB, pixelcount * sizeof(glm::vec3));
    hst_scene->state.bloom.assign(pixelcount, glm::vec3(0.0f));

    #if PROFILE
        cudaEventCreate(&evStart);
        cudaEventCreate(&evStop);
    #endif

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
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);

    cudaFree(dev_texPixels); dev_texPixels = NULL;
    cudaFree(dev_texInfo);   dev_texInfo = NULL;
    cudaFree(dev_lights);    dev_lights = NULL;

    cudaFree(dev_bloomA); dev_bloomA = NULL;
    cudaFree(dev_bloomB); dev_bloomB = NULL;

    cudaFree(dev_envMap);
    dev_envMap = NULL;

    cudaFree(dev_envCdf);
    dev_envCdf = NULL;

    #if PROFILE
    // pathtraceFree also runs before the first init, so guard against NULL
        if (evStart) { cudaEventDestroy(evStart); evStart = NULL; }
        if (evStop) { cudaEventDestroy(evStop);  evStop = NULL; }
    #endif

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

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jx = u01(rng) - 0.5f;
        float jy = u01(rng) - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jy - (float)cam.resolution.y * 0.5f)
        );

        if (cam.lensRadius > 0.0f) {
            // The point where the light from the pinhole strikes the focal plane
            float ft = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 pFocus = cam.position + ft * segment.ray.direction;

            // Select a random point uniformly on the lens disk
            float r = cam.lensRadius * sqrtf(u01(rng));
            float theta = TWO_PI * u01(rng);
            glm::vec3 lensPoint = cam.position
                + r * cosf(theta) * cam.right
                + r * sinf(theta) * cam.up;

            // Start from the point on the lens and aim at the focal point
            segment.ray.origin = lensPoint;
            segment.ray.direction = glm::normalize(pFocus - lensPoint);
        }


        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        segment.lastPdf = -1.0f;
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
    BVHNode* bvhNodes,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        // Guard: a NaN/Inf ray would visit every BVH node, so drop it here
        const Ray& rr = pathSegment.ray;
        if (!isfinite(rr.origin.x) || !isfinite(rr.origin.y) || !isfinite(rr.origin.z) ||
            !isfinite(rr.direction.x) || !isfinite(rr.direction.y) || !isfinite(rr.direction.z))
        {
            atomicAdd(&d_badRays, 1);
            intersections[path_index].t = -1.0f;
            pathSegments[path_index].color = glm::vec3(0.0f);
            pathSegments[path_index].remainingBounces = 0;
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool hit_outside = true;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv(0.0f), uv(0.0f);
        glm::vec3 tmp_tan(0.0f), tmp_bit(0.0f), hitTan(0.0f), hitBit(0.0f);

        // naive parse through global geoms

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
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, triangles, bvhNodes, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv, tmp_tan, tmp_bit);
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                hit_outside = outside;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = (geom.type == MESH) ? tmp_uv : glm::vec2(0.0f);
                hitTan = (geom.type == MESH) ? tmp_tan : glm::vec3(0.0f);
                hitBit = (geom.type == MESH) ? tmp_bit : glm::vec3(0.0f);
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
            intersections[path_index].uv = uv;
            intersections[path_index].tangent = hitTan;
            intersections[path_index].bitangent = hitBit;
            intersections[path_index].geomId = hit_geom_index;
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

// Look up the equirectangular environment map in direction dir
__device__ glm::vec3 sampleEnvironment(const glm::vec3* env, int w, int h, float rotation, glm::vec3 dir)
{
    dir = glm::normalize(dir);
    float phi = atan2f(dir.z, dir.x) + rotation;
    float theta = acosf(glm::clamp(dir.y, -1.0f, 1.0f));
    float u = phi / TWO_PI + 0.5f;
    u -= floorf(u);
    float v = theta / PI;
    int x = min(w - 1, (int)(u * w));
    int y = min(h - 1, (int)(v * h));
    return env[y * w + x];
}

// Bilinear texture lookup with repeat wrapping
__device__ glm::vec3 sampleTexture(const glm::vec3* pix, TextureInfo t, glm::vec2 uv)
{
    float u = uv.x - floorf(uv.x), v = uv.y - floorf(uv.y);
    float fx = u * t.width - 0.5f;
    float fy = (1.0f - v) * t.height - 0.5f;   // OBJ v points up, image rows go down
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float tx = fx - x0, ty = fy - y0;
    int x1 = (x0 + 1) % t.width;  x0 = (x0 + t.width) % t.width;
    int y1 = (y0 + 1) % t.height; y0 = (y0 + t.height) % t.height;
    const glm::vec3* p = pix + t.offset;
    glm::vec3 a = glm::mix(p[y0 * t.width + x0], p[y0 * t.width + x1], tx);
    glm::vec3 b = glm::mix(p[y1 * t.width + x0], p[y1 * t.width + x1], tx);
    return glm::mix(a, b, ty);
}

// Largest i with cdf[i] <= r
__device__ int sampleEnvCdf(const float* cdf, int n, float r) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (cdf[mid] <= r) lo = mid; else hi = mid - 1;
    }
    return min(lo, n - 1);
}

// Importance-sample a direction; outputs solid-angle pdf
__device__ glm::vec3 sampleEnvDirection(const float* cdf, int w, int h, float rotation,
    thrust::default_random_engine& rng, float& pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    int idx = sampleEnvCdf(cdf, w * h, u01(rng));
    float u = (idx % w + u01(rng)) / w;
    float v = (idx / w + u01(rng)) / h;
    float theta = v * PI;
    float phi = (u - 0.5f) * TWO_PI - rotation;
    float sinT = sinf(theta);
    pdf = sinT > 1e-6f ? (cdf[idx + 1] - cdf[idx]) * w * h / (2.0f * PI * PI * sinT) : 0.0f;
    return glm::vec3(sinT * cosf(phi), cosf(theta), sinT * sinf(phi));
}

// Pdf of sampleEnvDirection producing dir
__device__ float envPdf(const float* cdf, int w, int h, float rotation, glm::vec3 dir) {
    dir = glm::normalize(dir);
    float phi = atan2f(dir.z, dir.x) + rotation;
    float theta = acosf(glm::clamp(dir.y, -1.0f, 1.0f));
    float u = phi / TWO_PI + 0.5f; u -= floorf(u);
    int x = min(w - 1, (int)(u * w));
    int y = min(h - 1, (int)(theta / PI * h));
    int idx = y * w + x;
    float sinT = sinf(theta);
    return sinT > 1e-6f ? (cdf[idx + 1] - cdf[idx]) * w * h / (2.0f * PI * PI * sinT) : 0.0f;
}

// True if anything blocks the ray before distance maxT
__device__ bool isOccluded(Ray r, const Geom* geoms, int geomsSize,
    const Triangle* tris, const BVHNode* nodes, float maxT) {
    glm::vec3 p, nrm; bool outside;
    glm::vec2 uvDummy;
    glm::vec3 tDummy, bDummy;
    for (int i = 0; i < geomsSize; ++i) {
        const Geom& g = geoms[i];
        float t = -1.0f;
        if (g.type == CUBE) t = boxIntersectionTest(g, r, p, nrm, outside);
        else if (g.type == SPHERE) t = sphereIntersectionTest(g, r, p, nrm, outside);
        else if (g.type == MESH) t = meshIntersectionTest(g, tris, nodes, r, p, nrm, outside, uvDummy, tDummy, bDummy);
        if (t > 0.0f && t < maxT) return true;
    }
    return false;
}

// Uniformly sample a point on a cube or sphere light (by area)
__device__ void sampleLightPoint(const Geom& g, thrust::default_random_engine& rng,
    glm::vec3& p, glm::vec3& n) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float a = u01(rng) - 0.5f, b = u01(rng) - 0.5f;
    glm::vec3 lp, ln;
    if (g.type == CUBE) {
        // Pick a face with probability proportional to its area
        glm::vec3 s = g.scale;
        float ax = s.y * s.z, ay = s.x * s.z, az = s.x * s.y;
        float r = u01(rng) * (ax + ay + az);
        float side = u01(rng) < 0.5f ? -0.5f : 0.5f;
        if (r < ax) { lp = glm::vec3(side, a, b); ln = glm::vec3(side, 0, 0); }
        else if (r < ax + ay) { lp = glm::vec3(a, side, b); ln = glm::vec3(0, side, 0); }
        else { lp = glm::vec3(a, b, side); ln = glm::vec3(0, 0, side); }
    }
    else {
        // Uniform point on the unit sphere, radius 0.5 in object space
        float z = 1.0f - 2.0f * u01(rng);
        float rr = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        float phi = TWO_PI * u01(rng);
        ln = glm::vec3(rr * cosf(phi), rr * sinf(phi), z);
        lp = 0.5f * ln;
    }
    p = glm::vec3(g.transform * glm::vec4(lp, 1.0f));
    n = glm::normalize(glm::vec3(g.invTranspose * glm::vec4(ln, 0.0f)));
}

// 3D world-space checker, needs no UVs
__device__ glm::vec3 proceduralTexture(const Material& m, glm::vec3 p) {
    glm::vec3 q = glm::floor(p * m.procScale + 1e-4f);
    int parity = ((int)q.x + (int)q.y + (int)q.z) & 1;
    return parity ? m.procColor2 : m.color;
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    const glm::vec3* envMap,
    int envWidth,
    int envHeight,
    float envIntensity,
    float envRotation,
    const float* envCdf,
    const Geom* geoms,
    int geomsSize,
    const Triangle* tris,
    const BVHNode* nodes,
    glm::vec3* image,
    const glm::vec3* texPixels,
    const TextureInfo* texInfo,
    const int* lights,
    int numLights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& seg = pathSegments[idx];
    if (seg.remainingBounces <= 0) return;

    ShadeableIntersection isect = shadeableIntersections[idx];

    if (isect.t <= 0.0f) {
        // Escaped the scene: pick up light from the environment map, or black if there is none
        if (envMap != NULL) {
            glm::vec3 Le = envIntensity * sampleEnvironment(envMap, envWidth, envHeight, envRotation, seg.ray.direction);
#if ENV_NEE
            // MIS weight for BSDF-sampled hits on the env
            float pb = seg.lastPdf;
            if (envCdf != NULL && pb > 0.0f) {
                float pl = envPdf(envCdf, envWidth, envHeight, envRotation, seg.ray.direction);
                Le *= pb * pb / (pb * pb + pl * pl);
            }
#endif
            seg.color *= Le;
        }
        else
            seg.color = glm::vec3(0.0f);
        seg.remainingBounces = 0;
        return;
    }

    Material m = materials[isect.materialId];

    if (m.texId >= 0) m.color *= sampleTexture(texPixels, texInfo[m.texId], isect.uv);

    if (m.procTex == 1) {
        glm::vec3 hitP = getPointOnRay(seg.ray, isect.t);
        m.color = proceduralTexture(m, hitP);
    }

    // Normal mapping: tangent-space normal -> world space
    if (m.normalTexId >= 0) {
        glm::vec3 N = glm::normalize(isect.surfaceNormal);
        glm::vec3 T = isect.tangent - N * glm::dot(N, isect.tangent);   // Gram-Schmidt
        if (glm::dot(T, T) > 1e-12f) {
            T = glm::normalize(T);
            glm::vec3 B = glm::cross(N, T);
            if (glm::dot(B, isect.bitangent) < 0.0f) B = -B;           // mirrored UVs
            glm::vec3 tn = sampleTexture(texPixels, texInfo[m.normalTexId], isect.uv) * 2.0f - 1.0f;
            glm::vec3 Np = glm::normalize(tn.x * T + tn.y * B + tn.z * N);
            if (isfinite(Np.x)) isect.surfaceNormal = Np;
        }
    }

    if (m.emittance > 0.0f) {
        glm::vec3 Le = m.color * m.emittance;
#if LIGHT_NEE
        // MIS weight for BSDF-sampled hits on a light
        float pb = seg.lastPdf;
        const Geom& g = geoms[isect.geomId];
        if (numLights > 0 && pb > 0.0f && g.type != MESH) {
            float cosL = fabsf(glm::dot(isect.surfaceNormal, glm::normalize(seg.ray.direction)));
            float pl = isect.t * isect.t / (fmaxf(cosL, 1e-6f) * g.area * numLights);
            Le *= pb * pb / (pb * pb + pl * pl);
        }
#endif
        seg.color *= Le;
        seg.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg.remainingBounces);
    glm::vec3 hitPoint = getPointOnRay(seg.ray, isect.t);

#if ENV_NEE
    // Next event estimation toward the env map (diffuse only)
    if (envCdf != NULL && m.hasReflective == 0.0f && m.hasRefractive == 0.0f) {
        glm::vec3 nrm = isect.surfaceNormal;
        if (glm::dot(seg.ray.direction, nrm) > 0.0f) nrm = -nrm;
        float pl;
        glm::vec3 wi = sampleEnvDirection(envCdf, envWidth, envHeight, envRotation, rng, pl);
        float cosT = glm::dot(wi, nrm);
        if (pl > 0.0f && cosT > 0.0f) {
            Ray shadow;
            shadow.origin = hitPoint + nrm * 0.001f;
            shadow.direction = wi;
            if (!isOccluded(shadow, geoms, geomsSize, tris, nodes, FLT_MAX)) {
                float pb = cosT / PI;
                float w = pl * pl / (pl * pl + pb * pb);   // power heuristic
                glm::vec3 Le = envIntensity * sampleEnvironment(envMap, envWidth, envHeight, envRotation, wi);
                // One path per pixel per iteration, so no race here
                image[seg.pixelIndex] += seg.color * m.color * Le * (cosT / PI) * (w / pl);
            }
        }
    }
#endif

#if LIGHT_NEE
    // Next event estimation toward a random area light (diffuse only)
    if (numLights > 0 && m.hasReflective == 0.0f && m.hasRefractive == 0.0f) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 nrm = isect.surfaceNormal;
        if (glm::dot(seg.ray.direction, nrm) > 0.0f) nrm = -nrm;

        const Geom& L = geoms[lights[min((int)(u01(rng) * numLights), numLights - 1)]];
        glm::vec3 lp, ln;
        sampleLightPoint(L, rng, lp, ln);
        glm::vec3 d = lp - hitPoint;
        float dist2 = glm::dot(d, d);
        float dist = sqrtf(dist2);
        glm::vec3 wi = d / dist;
        float cosS = glm::dot(wi, nrm);
        float cosL = -glm::dot(wi, ln);
        if (cosS > 0.0f && cosL > 0.0f) {
            Ray shadow;
            shadow.origin = hitPoint + nrm * 0.001f;
            shadow.direction = wi;
            if (!isOccluded(shadow, geoms, geomsSize, tris, nodes, dist - 0.01f)) {
                // Area pdf -> solid angle: dist^2 / (cosL * area), times 1/numLights
                float pl = dist2 / (cosL * L.area * numLights);
                float pb = cosS / PI;
                float w = pl * pl / (pl * pl + pb * pb);
                const Material& lm = materials[L.materialid];
                image[seg.pixelIndex] += seg.color * m.color * (lm.color * lm.emittance) * (cosS / PI) * (w / pl);
            }
        }
    }
#endif

    scatterRay(seg, hitPoint, isect.surfaceNormal, isect.outside, m, rng);
    seg.remainingBounces--;

#if RUSSIAN_ROULETTE
    if (depth >= 3 && seg.remainingBounces > 0) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float pSurvive = glm::min(1.0f, glm::max(seg.color.r, glm::max(seg.color.g, seg.color.b)));
        if (u01(rng) >= pSurvive) {
            seg.color = glm::vec3(0.0f);
            seg.remainingBounces = 0;
            return;
        }
        seg.color /= pSurvive;
    }
#endif

    if (seg.remainingBounces == 0) {
        seg.color = glm::vec3(0.0f);
    }
}

struct isAlive
{
    __host__ __device__ bool operator()(const PathSegment& s) const
    {
        return s.remainingBounces > 0;
    }
};

#define SORT_BY_MATERIAL 0

struct MaterialIdLess
{
    __host__ __device__ bool operator()(const ShadeableIntersection& a,
        const ShadeableIntersection& b) const
    {
        return a.materialId < b.materialId;
    }
};

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        glm::vec3 c = iterationPath.color;
        // Drop NaN/Inf samples so a single bad path cannot stain a pixel
        if (isfinite(c.x) && isfinite(c.y) && isfinite(c.z))
            image[iterationPath.pixelIndex] += c;
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

        // Print alive paths per bounce on one chosen iteration
    bool logBounces = (iter == LOG_BOUNCE_ITER);
    auto countAlive = [&]() {
        return (int)thrust::count_if(thrust::device, dev_paths, dev_paths + num_paths, isAlive());
        };
    if (logBounces)
    {
        printf("--- alive paths per bounce (iter %d, compaction %s) ---\n",
            iter, STREAM_COMPACTION ? "on" : "off");
        int zero = 0;
        cudaMemcpyToSymbol(d_badRays, &zero, sizeof(int));
    }

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        if (logBounces) printf("bounce %d: %d\n", depth, countAlive());

        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        PROFILE_BEGIN();
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_bvhNodes,
            dev_intersections
            );
#if PROFILE
        {
            float ms = elapsedMs();
            timeIntersect += ms;
            if (logBounces)
            {
                int bad = 0, zero = 0;
                cudaMemcpyFromSymbol(&bad, d_badRays, sizeof(int));
                cudaMemcpyToSymbol(d_badRays, &zero, sizeof(int));
                printf("  intersect @ bounce %d: %.3f ms for %d paths, %d bad rays\n", depth, ms, num_paths, bad);
            }
        }
#endif
        checkCUDAError("trace one bounce");
        depth++;

#if SORT_BY_MATERIAL
        PROFILE_BEGIN();
        thrust::sort_by_key(thrust::device,
            dev_intersections, dev_intersections + num_paths,
            dev_paths,
            MaterialIdLess());
        PROFILE_END(timeSort);
#endif

        PROFILE_BEGIN();
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_envMap,
            hst_scene->envWidth,
            hst_scene->envHeight,
            hst_scene->envIntensity,
            hst_scene->envRotation,
            dev_envCdf,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_triangles,
            dev_bvhNodes,
            dev_image,
            dev_texPixels,
            dev_texInfo,
            dev_lights,
            (int)hst_scene->lights.size()
            );
        PROFILE_END(timeShade);

#if STREAM_COMPACTION
        PROFILE_BEGIN();
        PathSegment* alive_end = thrust::partition(
            thrust::device, dev_paths, dev_paths + num_paths, isAlive());
        num_paths = alive_end - dev_paths;
        PROFILE_END(timeCompact);
        iterationComplete = (num_paths == 0 || depth >= traceDepth);
#else
        // Without compaction dead paths stay in the array, so always run to max depth
        iterationComplete = (depth >= traceDepth);
#endif

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    if (logBounces) printf("bounce %d: %d\n", depth, countAlive());

#if PROFILE
    profiledIters++;
    if (profiledIters == PROFILE_WINDOW)
    {
        printf("[avg of %d iters] intersect %.3f | sort %.3f | shade %.3f | compact %.3f ms\n",
            PROFILE_WINDOW,
            timeIntersect / PROFILE_WINDOW, timeSort / PROFILE_WINDOW,
            timeShade / PROFILE_WINDOW, timeCompact / PROFILE_WINDOW);
        timeIntersect = timeSort = timeShade = timeCompact = 0.0;
        profiledIters = 0;
    }
#endif

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
        // Bloom: bright pass + separable Gaussian blur, composited at display time only
    const RenderState& rs = hst_scene->state;
    bool bloomOn = rs.bloomStrength > 0.0f;
    if (bloomOn)
    {
        brightPass << <numBlocksPixels, blockSize1d >> > (dev_image, dev_bloomA, pixelcount, iter, rs.bloomThreshold);
        float sigma = rs.bloomRadius / 3.0f;
        blurPass << <blocksPerGrid2d, blockSize2d >> > (dev_bloomA, dev_bloomB, cam.resolution.x, cam.resolution.y, rs.bloomRadius, sigma, true);
        blurPass << <blocksPerGrid2d, blockSize2d >> > (dev_bloomB, dev_bloomA, cam.resolution.x, cam.resolution.y, rs.bloomRadius, sigma, false);
    }
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, rs.toneMap,
        bloomOn ? dev_bloomA : NULL, rs.bloomStrength);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    if (bloomOn)
        cudaMemcpy(hst_scene->state.bloom.data(), dev_bloomA,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
