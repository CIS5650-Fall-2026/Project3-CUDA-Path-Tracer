#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
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
static Light* dev_lights = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static BVHNode* dev_bvhnodes = NULL;


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

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhnodes, scene->nodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhnodes, scene->nodes.data(), scene->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


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
    cudaFree(dev_bvhnodes);

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
        segment.radiance = glm::vec3(0.f);

        // jitter the ray for antialiasing effects
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float offsetx = u01(rng) - 0.5; // -0.5 to 0.5
        float offsety =  u01(rng) - 0.5; // -0.5 to 0.5

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + offsetx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + offsety - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.specularBounce = 1;
    }
}


__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Light* lights,
    int lights_size,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;
  
    PathSegment pathSegment = pathSegments[path_index];

    glm::vec3 geom_point(0.f);
    glm::vec3 geom_normal(0.f);
    bool geom_outside = true;
    int hit_geom_index = -1;

    // naive parse through sgeoms
    float t_geom = geometryIntersectionTest(
        geoms,
        geoms_size,
        bvhnodes,
        bvhnodes_size,
        pathSegment.ray,
        geom_point,
        geom_normal,
        geom_outside,
        hit_geom_index
    );


    glm::vec3 light_point(0.f);
    glm::vec3 light_normal(0.f);
    int hit_light_index = -1;

    float t_light = lightIntersectionTest(
        lights,
        lights_size,
        pathSegment.ray,
        light_point,
        light_normal,
        hit_light_index
    );

    bool geom_hit = hit_geom_index != -1 && t_geom > 0.f;
    bool light_hit = hit_light_index != -1 && t_light > 0.f;

    if (!geom_hit && !light_hit) // no intersection with anything
    {
        intersections[path_index].t = -1.f;
        intersections[path_index].isLight = 0;
        intersections[path_index].materialId = -1;
        intersections[path_index].lightId = -1;
        return;
    }

    if (light_hit && (!geom_hit || t_light < t_geom)) // light hit but no geom, or light hits first
    {
        intersections[path_index].t = t_light;
        intersections[path_index].isLight = 1;
        intersections[path_index].lightId = hit_light_index;
        intersections[path_index].materialId = -1;
        intersections[path_index].surfaceNormal = light_normal;
    }
    else // geom hit
    {
        intersections[path_index].t = t_geom;
        intersections[path_index].isLight = 0;
        intersections[path_index].materialId =
            geoms[hit_geom_index].materialid;
        intersections[path_index].lightId = -1;
        intersections[path_index].surfaceNormal = geom_normal;
    }
    
}


__global__ void shadeMaterial(
    int iter,int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    Light* lights,
    int lights_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];

    // intersection doesn't hit anything or behind camera
    if (intersection.t <= 0.0f) {
        pathSegment.remainingBounces = 0;
        return;
    }

   
    // If we actually hit a light, light the ray
    if (intersection.isLight) {
        if (intersection.lightId < 0 || intersection.lightId >= lights_size) {
            printf("Invalid light ID: %d, path: %d\n", intersection.lightId, idx);
            pathSegment.remainingBounces = 0;
            return;
        }
        Light light = lights[intersection.lightId];

        glm::vec3 contribution = light.color * light.intensity;

        if (light.type == AREALIGHT) {
            contribution /= (light.scale.x * light.scale.z);
        }

        // only count the light if direct lighting didn't already account for it
        if (pathSegment.specularBounce) {   // camera ray or glass bounce
            pathSegment.radiance += pathSegment.color * contribution;
        }

        pathSegment.remainingBounces = 0;
        return;
    }

    if (intersection.materialId < 0) {
        printf("Invalid material ID: %d, path: %d\n", intersection.materialId, idx);
        pathSegment.remainingBounces = 0;
        return;
    }
    Material material = materials[intersection.materialId];

    // If the material indicates that the object was a light, "light" the ray
    if (material.emittance > 0.0f) {
        glm::vec3 emission = material.color * material.emittance;
        pathSegment.radiance += pathSegment.color * emission;

        pathSegment.remainingBounces = 0;
        return;
    }

    if (pathSegment.remainingBounces <= 0) { // no contributionn if no more bounces
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

    glm::vec3 intersectPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

    // spawn new ray
    scatterRay(
        pathSegment,
        intersectPoint,
        intersection.surfaceNormal,
        material,
        geoms,
        geoms_size,
        bvhnodes,
        bvhnodes_size,
        lights,
        lights_size,
        rng
    );

    // after this, then our pathSegment should be completely updated here for the ray


    pathSegment.remainingBounces--;
}

__host__ __device__ glm::vec3 gammaReinhardt(glm::vec3 color)
{
    glm::vec3 outcol = color / (color + glm::vec3(1.f));
    outcol = glm::pow(outcol, glm::vec3(1.f / 2.2f));
    return outcol;

}

// Add the current iteration's output to the overall image
__global__ void finalGather(
    int nPaths,
    glm::vec3* image,
    PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces <= 0)
        {
            image[iterationPath.pixelIndex] += gammaReinhardt(iterationPath.radiance);
        }
    }
}
// helper for thrust::sort. Sorts the array based on materialID
struct sort_by_material {
    __host__ __device__ bool operator()(const thrust::tuple<ShadeableIntersection, PathSegment>& zipped_a, const thrust::tuple<ShadeableIntersection, PathSegment>& zipped_b) const
    {
        return thrust::get<0>(zipped_a).materialId < thrust::get<0>(zipped_b).materialId;// terminate if we don't intersect anything and if we are out of bounces
    }
};

//helper for thrust::removeif. Checks to see if intersection < 0. If so, terminate
struct terminateRays {
    __host__ __device__ bool operator()(const PathSegment& pathsegment) const
    {
        return pathsegment.remainingBounces <= 0;// terminate if we don't intersect anything and if we are out of bounces
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths); // set ray origin and ray dir, in dev_paths (pathsegments)
    // multiple iterations?

    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount; // end of the array
    int num_paths = dev_path_end - dev_paths; // number of pathsegments

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
            dev_lights,
            hst_scene->lights.size(),
            dev_bvhnodes,
            hst_scene->nodes.size(),
            dev_intersections
        );
        // dev_intersections should now be populated
        checkCUDAError("trace one bounce");
        cudaGetLastError();
        cudaDeviceSynchronize();

        depth++;

        // zip up with dev_paths, so the indices match
        auto dev_zipped = thrust::make_zip_iterator(thrust::make_tuple(dev_intersections, dev_paths));
        auto dev_zipped_end = thrust::make_zip_iterator(thrust::make_tuple(dev_intersections + num_paths, dev_paths + num_paths));

        // making contiguous in memory, sort by materialID
        thrust::sort(thrust::device, dev_zipped, dev_zipped_end, sort_by_material());

        // apply bsdf and populate color of paths
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_bvhnodes,
            hst_scene->nodes.size(),
            dev_lights,
            hst_scene->lights.size()
        );


        checkCUDAError("Shading material");

        // Stream compact away rays that don't intersect
        
        // copy color data to the image before we completely terminate the rays
        finalGather << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_image, dev_paths);
        // terminate rays that have no more bounces
        dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_path_end, terminateRays());
        num_paths = dev_path_end - dev_paths;
        
        if (num_paths == 0)
            iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
