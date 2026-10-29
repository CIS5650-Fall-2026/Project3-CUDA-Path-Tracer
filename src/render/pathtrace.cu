#include "render/pathtrace.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <thrust/random.h>

#include "scene/sceneStructs.h"
#include "scene/scene.h"
#include "glm/glm.hpp"
#include "timing.h"
#include "utilities.h"
#include "render/intersections.h"
#include "render/mesh_hit.h"
#include "render/bsdf.h"
#include "render/environment.h"
#include "render/lens_trace.h"
#include "render/exit_pupil.h"
#include "render/nee.h"
#include "render/pbr_surface.h"
#include "render/sampling.h"
#include "render/wavefront_ops.h"
#include "optix/optix_intersect.h"
#include "render/textures.h"

// Off in Release: the two device waits per bounce cost 5 to 20% of a sample.
// A kernel error still ends the run at wavefrontSwapPaths' stream wait.
#ifndef ERRORCHECK
#ifdef NDEBUG
#define ERRORCHECK 0
#else
#define ERRORCHECK 1
#endif
#endif

#define checkCUDAError(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError(), msg, file, line);
#endif // ERRORCHECK
}

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image,
    ToneMapMode toneMap, float exposure)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / (float)iter;
        pix = applyToneMap(pix, toneMap, exposure);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

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
static int* dev_materialIds = NULL;
// Swapped with the spare buffers in wavefront_ops.cu every bounce
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static MeshBuffers dev_mesh = {};
static cudaTextureObject_t* dev_textures = NULL;
static EnvironmentMap dev_environment = {};
static LightList dev_lights = {};
static LensSurface* dev_lens = NULL;
static float* dev_lensAspheric = NULL;  // not read yet
static LensDevice dev_lensDevice = {};
static glm::vec4* dev_lensPupil = NULL;
static float lensPupilMs = 0.0f;        // last exit pupil build
static float lensPupil0Area = 0.0f;     // the center pixel's pupil, mm^2: the simple weight's unit
static int* dev_lensBlocked = NULL;     // camera rays the lens blocked since the last reset
static double lensRays = 0.0;           // camera rays since the last reset
static void uploadLights();
static void freeLights();
static void uploadLens();
static void freeLens();

template <typename T>
static T* uploadVector(const std::vector<T>& v)
{
    if (v.empty())
    {
        return nullptr;
    }
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

void setGuiData(GuiDataContainer* data)
{
    guiData = data;
}

static bool useRussianRoulette = true;
void setRussianRoulette(bool enabled) { useRussianRoulette = enabled; }

static bool useAntialiasing = true;
void setAntialiasing(bool enabled) { useAntialiasing = enabled; }

static bool useMaterialSort = true;
void setMaterialSort(bool enabled) { useMaterialSort = enabled; }

static bool useCompaction = true;
void setCompaction(bool enabled) { useCompaction = enabled; }

static bool useExitPupil = true;
void setExitPupil(bool enabled) { useExitPupil = enabled; }

static bool usePhysicalCameraWeight = false;
void setPhysicalCameraWeight(bool enabled) { usePhysicalCameraWeight = enabled; }

static bool useNee = true;
void setNextEventEstimation(bool enabled) { useNee = enabled; }

static bool useOptix = true;
static bool optixValidation = false;
static bool optixReady = false;  // optixIntersectInit succeeded
void setOptix(bool enabled) { useOptix = enabled; }
void setOptixValidation(bool enabled) { optixValidation = enabled; }

bool pathtraceNeeAvailable() { return optixReady && dev_lights.numLights > 0; }

static ToneMapMode toneMapMode = TONEMAP_AGX_PUNCHY;
static float toneMapExposure = 1.f;
void setToneMap(ToneMapMode mode, float exposure)
{
    toneMapMode = mode;
    toneMapExposure = exposure;
}

// --timing: cudaEvents around every stage of every bounce, so a stage's time
// includes launch gaps and host synchronization inside it. Summed over iterations.
namespace
{
enum Stage { STAGE_INTERSECT, STAGE_SORT, STAGE_SHADE, STAGE_COMPACT, NUM_STAGES };
const char* const STAGE_NAMES[NUM_STAGES] = { "intersect", "sort", "shade", "compact" };

struct StageTiming
{
    int maxDepth = 0;
    int iterations = 0;
    double generateMs = 0.0;
    double shadowTailMs = 0.0;        // the last bounce's shadow ray launch, after the loop
    std::vector<double> stageMs;      // [depth * NUM_STAGES + stage]
    std::vector<double> alive;        // [depth]: paths entering bounce depth; [maxDepth]: left after the last
    std::vector<cudaEvent_t> events;  // 0 before generate, 1 after, then per stage, then after the shadow tail

    int eventIndex(int depth, int stage) const { return 2 + depth * NUM_STAGES + stage; }
    int tailIndex() const { return 2 + maxDepth * NUM_STAGES; }
    bool ready(int traceDepth) const { return !events.empty() && traceDepth <= maxDepth; }
} stageTiming;

void stageTimingFree()
{
    for (cudaEvent_t e : stageTiming.events)
    {
        cudaEventDestroy(e);
    }
    stageTiming = StageTiming();
}

void stageTimingInit(int maxDepth)
{
    stageTimingFree();
    stageTiming.maxDepth = maxDepth;
    stageTiming.stageMs.assign((size_t)maxDepth * NUM_STAGES, 0.0);
    stageTiming.alive.assign((size_t)maxDepth + 1, 0.0);
    stageTiming.events.resize(3 + (size_t)maxDepth * NUM_STAGES);
    for (cudaEvent_t& e : stageTiming.events)
    {
        CUDA_CHECK(cudaEventCreate(&e));
    }
}

void stageTimingAccumulate(int bounces)
{
    StageTiming& st = stageTiming;
    CUDA_CHECK(cudaEventSynchronize(st.events[st.tailIndex()]));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st.events[0], st.events[1]));
    st.generateMs += ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st.events[st.eventIndex(bounces - 1, STAGE_COMPACT)], st.events[st.tailIndex()]));
    st.shadowTailMs += ms;
    int previous = 1;
    for (int depth = 0; depth < bounces; ++depth)
    {
        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            const int current = st.eventIndex(depth, stage);
            CUDA_CHECK(cudaEventElapsedTime(&ms, st.events[previous], st.events[current]));
            st.stageMs[(size_t)depth * NUM_STAGES + stage] += ms;
            previous = current;
        }
    }
    ++st.iterations;
}
}  // namespace

void pathtraceTimingReport()
{
    const StageTiming& st = stageTiming;
    if (!timingEnabled() || st.iterations == 0)
    {
        return;
    }
    printf("TIMING,size.PathSegment_bytes,%zu\n", sizeof(PathSegment));
    printf("TIMING,size.ShadeableIntersection_bytes,%zu\n", sizeof(ShadeableIntersection));
    printf("TIMING,size.ShadowRay_bytes,%zu\n", sizeof(ShadowRay));
    printf("TIMING,scene.materials,%d\n", (int)hst_scene->materials.size());
    printf("TIMING,render.generate,%.4f\n", st.generateMs / st.iterations);
    if (dev_lensBlocked != nullptr && lensRays > 0.0)
    {
        printf("TIMING,lens.blocked_fraction,%.4f\n", pathtraceLensBlockedFraction());
        printf("TIMING,lens.exit_pupil_bins,%d\n", dev_lensDevice.pupilBinCount);
        printf("TIMING,lens.exit_pupil_ms,%.4f\n", lensPupilMs);
    }
    printf("TIMING,render.shadow_tail,%.4f\n", st.shadowTailMs / st.iterations);
    printf("BOUNCE,depth,alive_in,alive_out");
    for (int stage = 0; stage < NUM_STAGES; ++stage)
    {
        printf(",%s", STAGE_NAMES[stage]);
    }
    printf("\n");
    for (int depth = 0; depth < st.maxDepth; ++depth)
    {
        printf("BOUNCE,%d,%.1f,%.1f", depth + 1, st.alive[depth] / st.iterations, st.alive[depth + 1] / st.iterations);
        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            printf(",%.4f", st.stageMs[(size_t)depth * NUM_STAGES + stage] / st.iterations);
        }
        printf("\n");
    }
}

void pathtraceInit(Scene* scene)
{
    TimingScope timing("init.total");
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    size_t freeBefore = 0;
    size_t totalMemory = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBefore, &totalMemory));

    CUDA_CHECK(cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)));

    CUDA_CHECK(cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment)));

    dev_geoms = uploadVector(scene->geoms);
    dev_materials = uploadVector(scene->materials);
    uploadLens();

    uploadLights();

    CUDA_CHECK(cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection)));
    CUDA_CHECK(cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection)));

    CUDA_CHECK(cudaMalloc(&dev_materialIds, pixelcount * sizeof(int)));
    wavefrontInit(pixelcount);

    {
        TimingScope upload("init.mesh_upload");
        dev_mesh.positions = uploadVector(scene->positions);
        dev_mesh.normals = uploadVector(scene->normals);
        dev_mesh.uvs = uploadVector(scene->uvs);
        dev_mesh.tangents = uploadVector(scene->tangents);
        dev_mesh.indices = uploadVector(scene->indices);
        dev_mesh.meshes = uploadVector(scene->meshes);
    }

    {
        TimingScope upload("init.texture_upload");
        dev_textures = texturesInit(*scene);
        dev_environment = environmentInit(scene->environment);
    }

    optixReady = useOptix && optixIntersectInit(scene, dev_mesh, dev_materials, dev_textures, optixValidation);
    if (useOptix && !optixReady)
    {
        fprintf(stderr, "OptiX unavailable, using the naive intersection kernel\n");
    }
    if (useNee && !optixReady && dev_lights.numLights > 0)
    {
        fprintf(stderr, "Next event estimation needs OptiX for its shadow rays; lights are found by BSDF sampling only\n");
    }

    if (timingEnabled())
    {
        stageTimingInit(hst_scene->state.traceDepth);
        size_t freeAfter = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalMemory));
        timingAdd("init.gpu_footprint_mb", (double)(freeBefore - freeAfter) / (1024.0 * 1024.0));
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    freeLens();
    freeLights();
    cudaFree(dev_intersections);
    cudaFree(dev_materialIds);
    if (optixReady)
    {
        optixIntersectFree();  // before the mesh buffers it points into
        optixReady = false;
    }
    cudaFree(dev_mesh.positions);
    cudaFree(dev_mesh.normals);
    cudaFree(dev_mesh.uvs);
    cudaFree(dev_mesh.tangents);
    cudaFree(dev_mesh.indices);
    cudaFree(dev_mesh.meshes);
    dev_mesh = {};
    texturesFree();
    dev_textures = NULL;
    environmentFree();
    dev_environment = {};
    wavefrontFree();
    stageTimingFree();

    checkCUDAError("pathtraceFree");
}

static void uploadLens()
{
    const Scene& scene = *hst_scene;
    const Camera& cam = scene.state.camera;
    dev_lens = uploadVector(scene.lens.surfaces);
    dev_lensAspheric = uploadVector(scene.lens.aspheric);
    dev_lensDevice = {};
    if (dev_lens != nullptr)
    {
        const LensSystem& lens = scene.lens;
        const LensSystem::Paraxial& p = lens.paraxial;
        dev_lensDevice.surfaces = dev_lens;
        dev_lensDevice.count = (int)lens.surfaces.size();
        dev_lensDevice.entrancePupilZ = p.entrancePupilZ;
        dev_lensDevice.rearZ = p.filmDistance - lens.surfaces.back().thickness;
        // The film's height is the sensor's; its width follows the image
        dev_lensDevice.pixelMm = lens.sensorMm.y / (float)cam.resolution.y;
        dev_lensDevice.mmToScene = 0.001f * cam.unitsPerMeter;
        // The focus distance is measured from the entrance pupil, where cam.position sits
        dev_lensDevice.filmDistance = lensFilmDistance(lens, p.entrancePupilZ - cam.focusDistance / dev_lensDevice.mmToScene);
        const float filmWidth = lens.sensorMm.y * (float)cam.resolution.x / (float)cam.resolution.y;
        dev_lensDevice.halfDiagonal = 0.5f * sqrtf(filmWidth * filmWidth + lens.sensorMm.y * lens.sensorMm.y);
        dev_lensDevice.aimRadius = lensAimRadius(lens, dev_lensDevice.halfDiagonal);
        dev_lensDevice.pupilBinCount = exitPupilBinCount(dev_lensDevice.halfDiagonal);
        CUDA_CHECK(cudaMalloc(&dev_lensPupil, dev_lensDevice.pupilBinCount * sizeof(glm::vec4)));
        dev_lensDevice.pupil = dev_lensPupil;
        float* areas = nullptr;
        CUDA_CHECK(cudaMalloc(&areas, dev_lensDevice.pupilBinCount * sizeof(float)));
        lensPupilMs = exitPupilBuildDevice(dev_lensDevice, EXIT_PUPIL_GRID, dev_lensPupil, areas);
        CUDA_CHECK(cudaMemcpy(&lensPupil0Area, areas, sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(areas);
        CUDA_CHECK(cudaMalloc(&dev_lensBlocked, sizeof(int)));
        CUDA_CHECK(cudaMemset(dev_lensBlocked, 0, sizeof(int)));
        lensRays = 0.0;
    }
}

static void freeLens()
{
    cudaFree(dev_lens);
    dev_lens = NULL;
    cudaFree(dev_lensAspheric);
    dev_lensAspheric = NULL;
    cudaFree(dev_lensPupil);
    dev_lensPupil = NULL;
    cudaFree(dev_lensBlocked);
    dev_lensBlocked = NULL;
    dev_lensDevice = {};
}

void pathtraceSetLens()
{
    CUDA_CHECK(cudaDeviceSynchronize());
    freeLens();
    uploadLens();
}

void pathtraceSetEnvironment(const Environment& env)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    environmentFree();
    dev_environment = environmentInit(env);
    freeLights();
    hst_scene->buildLights();
    uploadLights();
}

void pathtraceReset()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    CUDA_CHECK(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)));
    if (dev_lensBlocked != nullptr)
    {
        CUDA_CHECK(cudaMemset(dev_lensBlocked, 0, sizeof(int)));
        lensRays = 0.0;
    }
}

int pathtraceExitPupilBins()
{
    return dev_lensDevice.pupilBinCount;
}

float pathtraceExitPupilBuildMs()
{
    return lensPupilMs;
}

float pathtraceLensFilmDistance()
{
    return dev_lensDevice.filmDistance;
}

float pathtraceLensBlockedFraction()
{
    if (dev_lensBlocked == nullptr || lensRays == 0.0)
    {
        return 0.0f;
    }
    int blocked = 0;
    CUDA_CHECK(cudaMemcpy(&blocked, dev_lensBlocked, sizeof(int), cudaMemcpyDeviceToHost));
    return (float)(blocked / lensRays);
}

static void uploadLights()
{
    dev_lights.triangles = uploadVector(hst_scene->lightTriangles);
    dev_lights.punctual = uploadVector(hst_scene->punctualLights);
    dev_lights.cdf = uploadVector(hst_scene->lightCdf);
    dev_lights.emitterAreaPdf = uploadVector(hst_scene->emitterAreaPdf);
    dev_lights.punctualPickPdf = uploadVector(hst_scene->punctualPickPdf);
    dev_lights.environmentPickPdf = uploadVector(hst_scene->environmentPickPdf);
    dev_lights.masks = uploadVector(hst_scene->lightMasks);
    dev_lights.numMasks = (int)hst_scene->lightMasks.size();
    dev_lights.numMaterials = (int)hst_scene->materials.size();
    dev_lights.numPunctual = (int)hst_scene->punctualLights.size();
    dev_lights.numTriangles = (int)hst_scene->lightTriangles.size();
    dev_lights.numLights = hst_scene->numLights;
}

static void freeLights()
{
    cudaFree((void*)dev_lights.triangles);
    cudaFree((void*)dev_lights.punctual);
    cudaFree((void*)dev_lights.cdf);
    cudaFree((void*)dev_lights.emitterAreaPdf);
    cudaFree((void*)dev_lights.punctualPickPdf);
    cudaFree((void*)dev_lights.environmentPickPdf);
    cudaFree((void*)dev_lights.masks);
    dev_lights = {};
}

// One path per pixel. The pinhole draws its two random numbers first and
// nothing else, so its image does not change when the other cameras draw
// more. A ray the lens blocks is black with no bounces left.
__global__ void generateRayFromCamera(Camera cam, LensDevice lens, int iter, int traceDepth, bool antialias,
    PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    bool blocked = false;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jx = u01(rng);
        float jy = u01(rng);
        if (!antialias)
        {
            jx = 0.5f;
            jy = 0.5f;
        }
        // Pixel (0, 0) is the top left: x runs along right, y against up
        const glm::vec3 d = cam.view
            + cam.right * cam.pixelLength.x * (float(x + jx) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + jy) - (float)cam.resolution.y * 0.5f);

        if (cam.type == CAMERA_THIN_LENS)
        {
            const glm::vec3 focus = cam.position + d * cam.focusDistance;
            const float r = cam.apertureRadius * sqrtf(u01(rng));
            const float phi = TWO_PI * u01(rng);
            segment.ray.origin = cam.position + cam.right * (r * cosf(phi)) + cam.up * (r * sinf(phi));
            segment.ray.direction = glm::normalize(focus - segment.ray.origin);
        }
        else if (cam.type == CAMERA_REAL_LENS && lens.surfaces != nullptr)
        {
            // The lens turns the image over, so the film point is mirrored in x and y
            const glm::vec3 film(
                -(float(x + jx) - (float)cam.resolution.x * 0.5f) * lens.pixelMm,
                ((float)(y + jy) - (float)cam.resolution.y * 0.5f) * lens.pixelMm,
                lens.filmDistance);
            const float u = u01(rng);
            const float v = u01(rng);
            const float rFilm = sqrtf(film.x * film.x + film.y * film.y);
            const float diskArea = PI * lens.aimRadius * lens.aimRadius;
            glm::vec4 box(0.0f);
            float area = diskArea;
            if (lens.useExitPupil)
            {
                const int bin = min((int)(rFilm / lens.halfDiagonal * (float)lens.pupilBinCount), lens.pupilBinCount - 1);
                box = lens.pupil[bin];
                area = (box.z - box.x) * (box.w - box.y);
            }
            glm::vec3 rear;
            if (area < diskArea)
            {
                // The bin's box, built for a film point on +x, turned to this film point's angle
                const float px = box.x + u * (box.z - box.x);
                const float py = box.y + v * (box.w - box.y);
                const float cosA = rFilm > 0.0f ? film.x / rFilm : 1.0f;
                const float sinA = rFilm > 0.0f ? film.y / rFilm : 0.0f;
                rear = glm::vec3(cosA * px - sinA * py, sinA * px + cosA * py, lens.rearZ);
            }
            else
            {
                // The whole rear element, when its disk is the smaller region around the pupil
                area = diskArea;
                const float r = lens.aimRadius * sqrtf(u);
                const float phi = TWO_PI * v;
                rear = glm::vec3(r * cosf(phi), r * sinf(phi), lens.rearZ);
            }
            glm::vec3 o = film;
            glm::vec3 dir = glm::normalize(rear - film);
            // cos^4 of the film-to-rear direction's angle to the axis, over the pdf of the sampled area
            const float cosTheta = -dir.z;
            const float cos2 = cosTheta * cosTheta;
            const float weight = cos2 * cos2 * area / lens.weightNorm;
            const bool out = area > 0.0f && traceLens(lens.surfaces, lens.count - 1, 0, lens.rearZ, o, dir)
                && isfinite(o.x + o.y + o.z + dir.x + dir.y + dir.z);
            if (out)
            {
                segment.color = glm::vec3(weight);
                // Lens z runs against view, with the entrance pupil at cam.position
                segment.ray.origin = cam.position
                    + (cam.right * o.x + cam.up * o.y + cam.view * (lens.entrancePupilZ - o.z)) * lens.mmToScene;
                segment.ray.direction = glm::normalize(cam.right * dir.x + cam.up * dir.y - cam.view * dir.z);
            }
            else
            {
                blocked = true;
                segment.color = glm::vec3(0.0f);
                segment.ray.origin = cam.position;
                segment.ray.direction = cam.view;
            }
        }
        else
        {
            segment.ray.origin = cam.position;
            segment.ray.direction = glm::normalize(d);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = blocked ? 0 : traceDepth;
        segment.medium = -1;
        segment.pdf = 0.0f;
        segment.receiverMask = 0xffffffffu;  // linking never hides an emitter from the camera
    }

    if (lens.blocked != nullptr)
    {
        // Nothing above returns early, so every lane votes
        const unsigned int votes = __ballot_sync(0xffffffffu, blocked);
        const int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31;
        if (lane == 0 && votes != 0u)
        {
            atomicAdd(lens.blocked, __popc(votes));
        }
    }
}

// The naive intersection stage: every path tests every Geom, closest hit wins.
__global__ void computeIntersections(
    int iter,
    int numPaths,
    PathSegment* pathSegments,
    int* materialIds,
    int numMaterials,
    Geom* geoms,
    int numGeoms,
    MeshBuffers buffers,
    const Material* materials,
    const cudaTextureObject_t* textures,
    ShadeableIntersection* intersections)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (pathIndex < numPaths)
    {
        PathSegment pathSegment = pathSegments[pathIndex];
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }
        const unsigned int alphaSeed = alphaPathSeed(iter, pathSegment.pixelIndex, pathSegment.remainingBounces);

        float t;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec4 tangent;
        float tMin = FLT_MAX;
        int hitGeomIndex = -1;
        bool closestOutside;

        glm::vec3 tmpPoint;
        glm::vec3 tmpNormal;
        glm::vec2 tmpUv;
        glm::vec4 tmpTangent;
        bool tmpOutside;

        for (int i = 0; i < numGeoms; i++)
        {
            Geom& geom = geoms[i];

            tmpUv = glm::vec2(0.0f);
            tmpTangent = glm::vec4(0.0f);
            if (geom.type == GeomType::CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmpPoint, tmpNormal, tmpOutside);
            }
            else if (geom.type == GeomType::SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmpPoint, tmpNormal, tmpOutside);
            }
            else
            {
                t = meshIntersectionTest(geom, i, buffers.meshes[geom.meshId], buffers, materials[geom.materialId],
                    textures, alphaSeed, pathSegment.ray, tmpPoint, tmpNormal, tmpUv, tmpTangent, tmpOutside);
            }

            if (t > 0.0f && tMin > t)
            {
                tMin = t;
                hitGeomIndex = i;
                normal = tmpNormal;
                uv = tmpUv;
                tangent = tmpTangent;
                closestOutside = tmpOutside;
            }
        }

        if (hitGeomIndex == -1)
        {
            intersections[pathIndex].t = -1.0f;
            materialIds[pathIndex] = numMaterials;  // misses sort last
        }
        else
        {
            intersections[pathIndex].t = tMin;
            intersections[pathIndex].materialId = geoms[hitGeomIndex].materialId;
            materialIds[pathIndex] = geoms[hitGeomIndex].materialId;
            intersections[pathIndex].surfaceNormal = normal;
            intersections[pathIndex].uv = uv;
            intersections[pathIndex].tangent = tangent;
            intersections[pathIndex].outside = closestOutside;
            intersections[pathIndex].lightMask = geoms[hitGeomIndex].lightMask;
            intersections[pathIndex].lightGroup = geoms[hitGeomIndex].lightGroup;
        }
    }
}

// OptiX when it initialized, else the naive kernel; OptiX also traces the queued shadow rays.
static void intersectScene(int iter, int numPaths, int numMaterials, const ShadowQueue& shadows, int shadowBound)
{
    if (optixReady)
    {
        optixIntersect(iter, numPaths, dev_paths, dev_intersections, dev_materialIds, numMaterials, shadows.rays,
            shadows.count, shadowBound, dev_image);
        return;
    }
    dim3 numBlocks = (numPaths + PATH_BLOCK_SIZE - 1) / PATH_BLOCK_SIZE;
    computeIntersections<<<numBlocks, PATH_BLOCK_SIZE>>>(
        iter,
        numPaths,
        dev_paths,
        dev_materialIds,
        numMaterials,
        dev_geoms,
        hst_scene->geoms.size(),
        dev_mesh,
        dev_materials,
        dev_textures,
        dev_intersections
    );
}

static_assert(PATH_BLOCK_SIZE % 32 == 0, "shadeMaterial's blocks must be whole warps");

// Shades every path, appends the survivors to survivors.paths (in place when
// compact is off) and, with nee, a shadow ray per hit to shadows.
__global__ void shadeMaterial(
    int iter,
    int numPaths,
    int depth,
    bool russianRoulette,
    bool nee,
    bool compact,
    ShadeableIntersection* shadeableIntersections,
    const PathSegment* pathSegments,
    SurvivorBuffer survivors,
    ShadowQueue shadows,
    Material* materials,
    const cudaTextureObject_t* textures,
    EnvironmentMap environment,
    LightList lights,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Every lane must reach the ballot at the end, so no early return. seg is
    // a register copy: scatterPbr takes it by reference.
    PathSegment seg;
    bool alive = false;
    if (idx < numPaths)
    {
        seg = pathSegments[idx];
        alive = seg.remainingBounces > 0;
    }
    const bool shaded = alive;
    ShadowRay shadow;
    bool queueShadow = false;
    if (alive)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];

        if (intersection.t <= 0.0f)
        {
            // A miss adds the environment, MIS-weighted against the environment sample the previous hit took
            if (environment.radiance != glm::vec3(0.0f))
            {
                float weight = 1.0f;
                const float environmentPick = nee && seg.pdf > 0.0f && lights.numLights > 0
                    ? lights.environmentPickPdf[lightRow(lights, seg.receiverMask)] : 0.0f;
                if (environmentPick > 0.0f)
                {
                    const float lightPdf = environmentPick * environmentPdf(environment, seg.ray.direction);
                    weight = powerHeuristic(seg.pdf, lightPdf);
                }
                image[seg.pixelIndex] += seg.color * environmentRadiance(environment, seg.ray.direction) * weight;
            }
            seg.remainingBounces = 0;
        }
        else
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, seg.pixelIndex, seg.remainingBounces);
            const Material& material = materials[intersection.materialId];
            const glm::vec3 hitPoint = seg.ray.origin + intersection.t * seg.ray.direction;
            const glm::vec3 wo = -seg.ray.direction;

            // Beer-Lambert absorption along the segment inside a volume
            if (seg.medium >= 0)
            {
                const glm::vec3 absorption = materials[seg.medium].absorption;
                if (absorption != glm::vec3(0.0f))
                {
                    seg.color *= glm::exp(-absorption * intersection.t);
                }
            }

            const bool lastBounce = seg.remainingBounces == 1;
            PbrSurface surface;
            glm::vec3 emission;
            if (lastBounce)
            {
                emission = pbrEmission(material, intersection, wo, textures);
            }
            else
            {
                surface = pbrSurface(material, intersection, wo, textures);
                emission = surface.emission;
            }
            // Emission, MIS-weighted against the light sample the previous hit
            // took (pdf 0: camera ray, smooth lobe, or an emitter NEE does not sample)
            if (emission != glm::vec3(0.0f) && receivesGroup(seg.receiverMask, intersection.lightGroup))
            {
                float weight = 1.0f;
                if (nee && seg.pdf > 0.0f)
                {
                    const float areaPdf = lights.emitterAreaPdf[lightRow(lights, seg.receiverMask) * lights.numMaterials
                        + intersection.materialId];
                    if (areaPdf > 0.0f)
                    {
                        const float lightPdf = areaPdf * intersection.t * intersection.t / intersection.cosGeometric;
                        weight = powerHeuristic(seg.pdf, lightPdf);
                    }
                }
                image[seg.pixelIndex] += seg.color * emission * weight;
            }
            seg.receiverMask = intersection.lightMask;

            if (lastBounce)
            {
                seg.remainingBounces = 0;
            }
            else
            {
                // One light sample per hit, before Russian roulette as in PBRT
                if (nee)
                {
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    const float u0 = u01(rng);
                    const float u1 = u01(rng);
                    const float u2 = u01(rng);
                    LightSample light;
                    if (sampleLight(lights, environment, materials, textures, hitPoint, intersection.lightMask, u0, u1,
                            u2, light))
                    {
                        const BsdfEval f = evalPbr(wo, light.wi, intersection.surfaceNormal, intersection.outside,
                            material, surface);
                        if (f.fCos != glm::vec3(0.0f))
                        {
                            const float weight = light.pdf > 0.0f ? powerHeuristic(light.pdf, f.pdf) : 1.0f;
                            const float side = glm::dot(light.wi, intersection.surfaceNormal) > 0.0f ? 1.0f : -1.0f;
                            shadow.origin = offsetOrigin(hitPoint, side * intersection.surfaceNormal);
                            shadow.direction = light.wi;
                            shadow.tMax = light.tMax;
                            shadow.contribution = seg.color * f.fCos * light.weightedLight * weight;
                            shadow.pixelIndex = seg.pixelIndex;
                            shadow.alphaSeed = shadowAlphaSeed(iter, seg.pixelIndex, seg.remainingBounces);
                            queueShadow = true;
                        }
                    }
                }

                if (russianRoulette && depth >= 3)
                {
                    const glm::vec3 c = seg.color;
                    const float p = glm::min(0.95f, 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b);
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    if (u01(rng) >= p)
                    {
                        seg.remainingBounces = 0;
                    }
                    else
                    {
                        seg.color /= p;
                    }
                }

                if (seg.remainingBounces > 0)
                {
                    if (scatterPbr(seg, hitPoint, intersection.surfaceNormal, intersection.outside, material, surface,
                            rng))
                    {
                        --seg.remainingBounces;
                        // A direction past the surface refracted into the object or out of it
                        if (glm::dot(seg.ray.direction, intersection.surfaceNormal) < 0.0f)
                        {
                            seg.medium = intersection.outside ? intersection.materialId : -1;
                        }
                    }
                    else
                    {
                        seg.remainingBounces = 0;
                    }
                }
            }
        }

        alive = seg.remainingBounces > 0;
    }

    // Warp-aggregated append: lane 0 reserves the warp's slots with one atomicAdd
    const unsigned int warpSurvivors = __ballot_sync(0xffffffffu, alive);
    const unsigned int lane = threadIdx.x % 32;
    int base = 0;
    if (lane == 0 && warpSurvivors != 0)
    {
        base = atomicAdd(survivors.count, __popc(warpSurvivors));
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (!compact)
    {
        if (shaded)
        {
            survivors.paths[idx] = seg;
        }
    }
    else if (alive)
    {
        const unsigned int lanesBelow = (1u << lane) - 1u;
        survivors.paths[base + __popc(warpSurvivors & lanesBelow)] = seg;
    }

    if (idx == 0)
    {
        *survivors.nextCount = 0;
    }

    if (nee)
    {
        const unsigned int warpShadows = __ballot_sync(0xffffffffu, queueShadow);
        int shadowBase = 0;
        if (lane == 0 && warpShadows != 0)
        {
            shadowBase = atomicAdd(shadows.count, __popc(warpShadows));
        }
        shadowBase = __shfl_sync(0xffffffffu, shadowBase, 0);
        if (queueShadow)
        {
            shadows.rays[shadowBase + __popc(warpShadows & ((1u << lane) - 1u))] = shadow;
        }
        if (idx == 0)
        {
            *shadows.nextCount = 0;
        }
    }
}

static int bitsToHold(int maxValue)
{
    int bits = 1;
    while ((1 << bits) <= maxValue)
    {
        ++bits;
    }
    return bits;
}

void pathtrace(uchar4* pbo, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int numMaterials = (int)hst_scene->materials.size();
    const int materialKeyBits = bitsToHold(numMaterials);  // numMaterials is the miss key

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const bool timing = timingEnabled() && stageTiming.ready(traceDepth);
    if (timing)
    {
        CUDA_CHECK(cudaEventRecord(stageTiming.events[0]));
    }
    LensDevice lens = dev_lensDevice;
    lens.blocked = cam.type == CAMERA_REAL_LENS && dev_lens != nullptr ? dev_lensBlocked : nullptr;
    lens.useExitPupil = useExitPupil ? 1 : 0;
    const float filmToRear = lens.filmDistance - lens.rearZ;
    lens.weightNorm = usePhysicalCameraWeight ? filmToRear * filmToRear : lensPupil0Area;
    if (lens.blocked != nullptr)
    {
        lensRays += pixelcount;
    }
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, lens, iter, traceDepth, useAntialiasing, dev_paths);
    checkCUDAError("generate camera ray");
    if (timing)
    {
        CUDA_CHECK(cudaEventRecord(stageTiming.events[1]));
    }

    int depth = 0;
    int numPaths = pixelcount;  // paths the launches cover; differs from numAlive only without compaction
    int numAlive = pixelcount;

    // Shadow rays are traced one bounce late, by the next intersection launch
    const bool nee = useNee && pathtraceNeeAvailable();
    ShadowQueue queuedShadows = {};
    int shadowBound = 0;

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        if (timing)
        {
            stageTiming.alive[depth] += numAlive;
        }

        dim3 numBlocks = (numPaths + PATH_BLOCK_SIZE - 1) / PATH_BLOCK_SIZE;
        intersectScene(iter, numPaths, numMaterials, queuedShadows, shadowBound);
        checkCUDAError("trace one bounce");
#if ERRORCHECK
        cudaDeviceSynchronize();
#endif
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_INTERSECT)]));
        }

        if (useMaterialSort)
        {
            sortMaterials(numPaths, materialKeyBits, dev_materialIds, dev_intersections, dev_paths);
        }
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_SORT)]));
        }

        shadeMaterial<<<numBlocks, PATH_BLOCK_SIZE>>>(
            iter,
            numPaths,
            depth + 1,
            useRussianRoulette,
            nee,
            useCompaction,
            dev_intersections,
            dev_paths,
            useCompaction ? wavefrontSurvivors() : wavefrontSurvivorsInPlace(dev_paths),
            wavefrontShadowQueue(),
            dev_materials,
            dev_textures,
            dev_environment,
            dev_lights,
            dev_image
        );
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_SHADE)]));
        }
        shadowBound = 0;
        if (nee)
        {
            queuedShadows = wavefrontSwapShadowQueue();
            shadowBound = depth + 1 < traceDepth ? numPaths : 0;  // the last bounce queues nothing
        }
        if (useCompaction)
        {
            numAlive = wavefrontSwapPaths(dev_paths);
            numPaths = numAlive;
        }
        else
        {
            numAlive = wavefrontCountSurvivors();
        }
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_COMPACT)]));
        }
        depth++;

        iterationComplete = numAlive == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->tracedDepth = depth;
        }
    }

    // The last bounce's shadow rays, with no paths left to trace alongside
    if (shadowBound > 0)
    {
        optixIntersect(iter, 0, dev_paths, dev_intersections, dev_materialIds, numMaterials, queuedShadows.rays,
            queuedShadows.count, shadowBound, dev_image);
        checkCUDAError("trace the last shadow rays");
    }
    if (timing)
    {
        CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.tailIndex()]));
        stageTiming.alive[depth] += numAlive;
        stageTimingAccumulate(depth);
    }

    if (pbo != nullptr)
    {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image,
            toneMapMode, toneMapExposure);
    }

    checkCUDAError("pathtrace");
}

void pathtraceDownloadImage()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    CUDA_CHECK(cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
}
