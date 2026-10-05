#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cuda/std/functional>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

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
unsigned long long makeSeededRandomEngine(int iter, int index, int depth)
{
    unsigned long long state = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    if (state == 0) state = 1;
    return state;
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
static int* dev_path_flags = NULL;
static Geom* dev_light_sources = NULL;
static int light_sources_count = 0;


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
    cudaMalloc(&dev_path_flags, pixelcount * sizeof(int));

    std::vector<Geom> lights;
    for (const Geom& obj : scene->geoms) {
        if (scene->materials[obj.materialid].emittance > 0.0f) {
            lights.push_back(obj);
        }
    }
    light_sources_count = (int)lights.size();
    cudaMalloc(&dev_light_sources, light_sources_count * sizeof(Geom));
    cudaMemcpy(dev_light_sources, lights.data(), light_sources_count * sizeof(Geom), cudaMemcpyHostToDevice);



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
    cudaFree(dev_path_flags);
    cudaFree(dev_light_sources);

    checkCUDAError("pathtraceFree");
}






/**
* Here I added my personal extra credit and all of the helper methods needed for it that was approved in Ed discussions.
*
* It is a mirage effect when a ray passes through material with variable index of refraction
* Good example of that is a hot air. Its IOR is affected by temperature, so the hotter the air, the bigger the difference.
*
* I think it is interesting and prettty cool physical effect and it is especially nice
* when sometimes in the modern games I also see this effect present. To me it means they really pay close attention to details.
* This is a pretty simple version and I am sure BigTech companies have much better logic for this, but here is my implementation.
* Hope you enjoy :)
*/


// our hot air box structure
struct HotAirBox
{
    glm::vec3 bmin;
    glm::vec3 bmax;
    float bottomTemperatureC;
    float topTemperatureC;
    float visualStrength; // 1 = "how it is in real world" and >1 is for bigger effect
};


// checking if position of the ray is insede hot air box (do we even need to bend it)
__device__ bool insideHotAir(glm::vec3 pos, HotAirBox box)
{
    return pos.x >= box.bmin.x && pos.x <= box.bmax.x
        && pos.y >= box.bmin.y && pos.y <= box.bmax.y
        && pos.z >= box.bmin.z && pos.z <= box.bmax.z;
}

// finding index of refraction at a specific pos since it is a distribution (from super hot at the bottom of the box to room temperature at the top)
__device__ float hotAirIOR(glm::vec3 pos, HotAirBox box)
{
    float heightRatio  = (pos.y - box.bmin.y) / (box.bmax.y - box.bmin.y + 1e-5f); // try to find ratio in terms of height
    heightRatio = glm::clamp(heightRatio, 0.0f, 1.0f);

    float temperatureC = box.bottomTemperatureC * (1.0f - heightRatio) + box.topTemperatureC * heightRatio; // temperature at our position
    float temperatureK = temperatureC + 273.15f; // physical formula uses Kelvin

    return 1.0f + (0.000293f * 273.15f) / temperatureK; // 0.000293 is the physical constant for air
}

__device__ void marchBendRay(Ray& ray, HotAirBox box)
{
    const float stepSize = 0.12f; // distance we travel in each step, not something physics-related
    const int maxSteps = 48;

    glm::vec3 pos = ray.origin;
    glm::vec3 dir = glm::normalize(ray.direction);

    // if we start outside, step forward a bit until we hit the box or give up
    if (!insideHotAir(pos, box)) {
        for (int i = 0; i < 64; i++) {
            pos += dir * stepSize;
            if (insideHotAir(pos, box)) break;
        }
        if (!insideHotAir(pos, box)) return;
    }

    const float gradientStep = 0.05f; // interval to estimate IOR, not physics-related eitherr
    for (int i = 0; i < maxSteps; i++)
    {
        if (!insideHotAir(pos, box)) break;


        float yBelow = glm::max(pos.y - gradientStep, box.bmin.y);
        float yAbove = glm::min(pos.y + gradientStep, box.bmax.y);


        // Central Difference Scheme (this is very common in numerical analysis if you are familiar with it)
        float n0 = hotAirIOR(pos, box);
        float nBelow = hotAirIOR(glm::vec3(pos.x, yBelow, pos.z), box);
        float nAbove = hotAirIOR(glm::vec3(pos.x, yAbove, pos.z), box);
        

        float gradientDistance = yAbove - yBelow;
        if (gradientDistance <= 1e-6f) break; // make sure we dont divide by zero

        glm::vec3 grad(0.0f, (nAbove - nBelow) / gradientDistance, 0.0f);


        glm::vec3 turn = grad - dir * glm::dot(dir, grad); // this is sort of an adaptation of Snells law but for continuous media
        dir = glm::normalize(dir + box.visualStrength * stepSize * turn / n0);

        glm::vec3 nextPos = pos + dir * stepSize;
        if (!insideHotAir(nextPos, box)) break; // if we are out of the box, leave the loop
        pos = nextPos;
    }

    ray.origin = pos;
    ray.direction = dir;
}

__global__ void applyHotAirMarch(int num_paths, PathSegment* pathSegments)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    if (pathSegments[idx].remainingBounces <= 0) return;

    // Slab just above the floor, you can play around with parameters to place it somewhere else
    HotAirBox box;
    box.bmin = glm::vec3(-4.5f, 0.05f, -4.5f);
    box.bmax = glm::vec3(4.5f, 1.8f, 4.5f);
    box.bottomTemperatureC = 800.0f;
    box.topTemperatureC = 20.0f;
    box.visualStrength = 0.0f; // to see some real effect put around 200
                                 // keep in mind that irl distances are much larger, so here we need to scale up a lot
    
    marchBendRay(pathSegments[idx].ray, box);
}


/**
* End of my personal extra credit :) 
*/






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
        unsigned long long rng = makeSeededRandomEngine(iter, index, 0);
        float x_jitter = betterRandom01(rng) - 0.5f;
        float y_jitter = betterRandom01(rng) - 0.5f;

        // Propaerties of our lens
        float focalDistance = 8.0f;
        float lensSize = 0.0f;

        glm::vec3 pinholeDirection = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + x_jitter)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + y_jitter));


        glm::vec3 pixelFocalPoint = cam.position + pinholeDirection * focalDistance;

        // Polar coordinates for the lens
        float lensRadius = sqrt(betterRandom01(rng)) * lensSize;
        float lensTheta = betterRandom01(rng) * TWO_PI;

        // Offset back to cartesian coordinates
        glm::vec3 lensOffset = lensRadius * cos(lensTheta) * cam.right + lensRadius * sin(lensTheta) * cam.up;
        
        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction = glm::normalize(pixelFocalPoint - segment.ray.origin);

        

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

__device__ bool somethingOnTheWay( 
    glm::vec3 from,
    glm::vec3 to,
    Geom* geoms,
    int geoms_size,
    Material* materials)
{
    glm::vec3 direction = glm::normalize(to - from);
    float distance = length(to - from);

    Ray ray;
    ray.origin = from + direction * 0.001f;
    ray.direction = direction;

    float t;
    float t_min = distance - 0.001f;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    bool hit = false;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
        }
        else{
            continue;
        }
        

        // like in computeIntersections but simplier for true/false indication
        if (t > 0.0f && t_min > t)
        {
            if (materials[geom.materialid].emittance > 0.0f) continue;
            hit = true;
            break;
        }
    }

    return hit;


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
    Material* materials,
    int depth,
    int numberOfBounces,
    glm::vec3* image,
    Geom* geoms,
    int geoms_size,
    Geom* light_sources,
    int light_sources_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces <= 0){
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            unsigned long long rng = makeSeededRandomEngine(iter, idx, depth);

            Material material = materials[intersection.materialId];

            glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;


            #if 1
                if (material.emittance <= 0.0f && material.hasRefractive <= 0.0f && light_sources_count > 0)
                {
                    int lightID = (int)(betterRandom01(rng) * light_sources_count);
                    if (lightID >= light_sources_count) lightID = light_sources_count - 1; // if (very unlikely) betterRandom01 will spit out 1



                    Geom light_source = light_sources[lightID];
                    Material light_source_material = materials[light_source.materialid];

                    glm::vec3 local(betterRandom01(rng) - 0.5f, -0.5f, betterRandom01(rng) - 0.5f); // a point on light where x and z are random and y is bottom
                    glm::vec3 pointOnSource = glm::vec3(light_source.transform * glm::vec4(local, 1.0f)); // cooridinates in the room


                    glm::vec3 lightVector = pointOnSource - intersect;
                    float dist = glm::length(lightVector);
                    glm::vec3 lightDirection = lightVector / dist;
                    
                    float cosLighting = glm::dot(intersection.surfaceNormal, lightDirection);
                    if (cosLighting > 0.0f && !somethingOnTheWay(intersect, pointOnSource, geoms, geoms_size, materials))
                    {
                        float totalLight = light_source.scale.x * light_source.scale.z; // area of one face
                        glm::vec3 directLightEffect = pathSegments[idx].color * material.color * light_source_material.color * light_source_material.emittance * cosLighting * totalLight / (dist * dist);
                        image[pathSegments[idx].pixelIndex] += directLightEffect / 3.1415f;
                    }

                
                }
            #endif

            scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);

            #if 1
                // if we think ray might be dark, lets play russian roulette
                if (pathSegments[idx].remainingBounces > 0 && pathSegments[idx].remainingBounces <= numberOfBounces/2)
                {

                    float highestColorBrightness = glm::max(pathSegments[idx].color.x, pathSegments[idx].color.y);
                    highestColorBrightness = glm::max(highestColorBrightness, pathSegments[idx].color.z);

                    float rouletteChances = glm::clamp(1.0f - highestColorBrightness, 0.0f, 5.0f / 6.0f); // extra fun roulette, only 1 empty chamber

                    if (betterRandom01(rng) < rouletteChances) // the ray was unlucky and died
                    {
                        pathSegments[idx].remainingBounces = 0;
                        pathSegments[idx].color = glm::vec3(0.0f);
                    }
                    else // the ray was lucky and survived
                    {
                        pathSegments[idx].color /= (1.0f - rouletteChances); // even out the color to account for those who died
                    }
                }
            #endif


            if (pathSegments[idx].remainingBounces <= 0) {
                image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
                pathSegments[idx].color = glm::vec3(0.0f);
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

__global__ void markDeadPaths(int nPaths, PathSegment* paths, int* flags)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nPaths)
    {
        flags[index] = paths[index].remainingBounces <= 0 ? 1 : 0;
    }
}

__global__ void writeMaterialIds(int nPaths, ShadeableIntersection* intersections, int* keys)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nPaths)
    {
        keys[index] = intersections[index].t > 0.0f ? intersections[index].materialId : -1; // -1 is for those that flew out of the room
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

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // my hot air feature
        applyHotAirMarch<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_paths
        );
        checkCUDAError("hot air march");

        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        if (guiData != NULL && guiData->materialSorting)
        {
            writeMaterialIds<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_path_flags);
            checkCUDAError("material ids");

            auto zipped = thrust::make_zip_iterator(thrust::device_pointer_cast(dev_paths),
                thrust::device_pointer_cast(dev_intersections));
            
            thrust::sort_by_key(thrust::device,thrust::device_pointer_cast(dev_path_flags),
                thrust::device_pointer_cast(dev_path_flags) + num_paths,zipped);
        }

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
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth,
            traceDepth,
            dev_image,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_light_sources,
            light_sources_count
        );

        checkCUDAError("shade");

        markDeadPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_paths, dev_path_flags);
        checkCUDAError("mark dead paths");

        thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
        thrust::device_ptr<int> thrust_flags(dev_path_flags);
        thrust::device_ptr<PathSegment> new_end = thrust::remove_if(thrust_paths, thrust_paths + num_paths, thrust_flags, cuda::std::identity{});
        
        num_paths = static_cast<int>(new_end - thrust_paths);

        iterationComplete = (depth >= traceDepth) || (num_paths == 0);

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
