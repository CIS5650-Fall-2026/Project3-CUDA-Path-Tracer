#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>  
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"
#include "denoiser.h"

#define ERRORCHECK 0
#define STREAM_COMPACTION 1
#define MATERIAL_SORTING 0
#define USE_BVH 1
#define USE_DENOISER 0
#define FOCAL_DISTANCE 20
#define APERTURE_RADIUS 0.15

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

// Kernel that writes the image to the OpenGL PBO directly.
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
static Triangle* dev_triangles = NULL;
static Geom* dev_lights = NULL;
static BVHNode* dev_nodes = NULL;
static int* dev_triPtrs = NULL;

static cudaArray* dev_mapdata = NULL;
static cudaTextureObject_t dev_envmap = NULL;

static cudaTextureObject_t* dev_textures = NULL;

// so i can free later 
static std::vector<cudaArray_t> host_texData;
static std::vector<cudaTextureObject_t> host_textures;

static glm::vec3* dev_avgImg = NULL;
static glm::vec3* dev_denoised = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    // build BVH
    BVH bvh;
    bvh.buildBVH(scene);

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_avgImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_avgImg, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_nodes, bvh.nodesUsed * sizeof(BVHNode));
    cudaMemcpy(dev_nodes, bvh.bvhNodePool.data(), bvh.nodesUsed * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triPtrs, scene->triangles.size() * sizeof(int));
    cudaMemcpy(dev_triPtrs, bvh.triIdx.data(), scene->triangles.size() * sizeof(int), cudaMemcpyHostToDevice);

    // for environment map 
    if (scene->envMapPixels.size() > 0) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        cudaMallocArray(&dev_mapdata, &channelDesc, scene->envMapWidth, scene->envMapHeight);
        cudaMemcpy2DToArray(
            dev_mapdata, 0, 0,
            scene->envMapPixels.data(),
            scene->envMapWidth * sizeof(float4),
            scene->envMapWidth * sizeof(float4),
            scene->envMapHeight,
            cudaMemcpyHostToDevice
        );

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_mapdata;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&dev_envmap, &resDesc, &texDesc, nullptr);
    }

    // for textures 
    if (scene->textures.size() > 0) {
        for (int i = 0; i < scene->textures.size(); ++i) {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

            std::vector<uint8_t>& texture = scene->textures[i];
            int texWidth = scene->texDims[i][0];
            int texHeight = scene->texDims[i][1];

            cudaArray_t dev_texData;
            cudaMallocArray(&dev_texData, &channelDesc, texWidth, texHeight);
            cudaMemcpy2DToArray(
                dev_texData, 0, 0,
                texture.data(),
                texWidth * sizeof(uchar4),
                texWidth * sizeof(uchar4),
                texHeight,
                cudaMemcpyHostToDevice
            );

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = dev_texData;

            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = 1;

            cudaTextureObject_t dev_texture;
            cudaCreateTextureObject(&dev_texture, &resDesc, &texDesc, nullptr);

            host_texData.push_back(dev_texData);
            host_textures.push_back(dev_texture);
        }

        cudaMalloc(&dev_textures, host_textures.size() * sizeof(cudaTextureObject_t));
        cudaMemcpy(dev_textures, host_textures.data(), host_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }

    // initialize denoiser
    initDenoiser(cam.resolution.x, cam.resolution.y, dev_image, dev_denoised);
    
    checkCUDAError("pathtraceInit");
}

void pathtraceReset() {
    // just want to reset the camera and scene!
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceReset");
}

void pathtraceFree()
{
    cudaFree(dev_image); 
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_lights);
    cudaFree(dev_triangles);
    cudaFree(dev_nodes);
    cudaFree(dev_triPtrs);

    cudaFree(dev_mapdata);
    cudaDestroyTextureObject(dev_envmap);

    for (auto texture : host_textures) {
        cudaDestroyTextureObject(texture);
    }
    host_textures.clear();

    for (auto data : host_texData) {
        cudaFree(data);
    }
    host_texData.clear();

    cudaFree(dev_textures);

    cudaFree(dev_avgImg);
    cudaFree(dev_denoised);
    denoiserFree();

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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);
        
        float offsetX = u01(rng) - 0.5f;
        float offsetY = u01(rng) - 0.5f;
        
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x  + offsetX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + offsetY - (float)cam.resolution.y * 0.5f)
        );

        // where does ray hit focal plane (defined by focal distance) 
        float t = FOCAL_DISTANCE / glm::dot(segment.ray.direction, cam.view);
        glm::vec3 focalPoint = cam.position + segment.ray.direction * t;

        // pick random spot on aperture 
        float r = APERTURE_RADIUS * sqrt(u01(rng));
        float theta = 2.0f * PI * sqrt(u01(rng));
        float x = r * cos(theta); 
        float y = r * sin(theta);
        glm::vec3 apertureOffset = cam.right * x + cam.up * y;

        // shoot ray from aperture point to focal point 
        segment.ray.origin = cam.position + apertureOffset;
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

/**
* Compute intersections and store information for shading.
*
* Handles geometry intersetions (sphere, cube)
* Handles triangle intersections for arbitrary meshes using BVH
*/
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    int triangles_size,
    ShadeableIntersection* intersections,
    BVHNode* nodes,
    int* triPtrs)
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
        bool tmp_outside;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = outside;
        }

        // traverse through BVH to find triangle intersections 

        if (triangles_size != 0) {
            int hit_tri_index = -1;

            float u;
            float v;

            #if USE_BVH
            IntersectBVH(pathSegment.ray, nodes, triangles, triPtrs, t_min, intersect_point, normal, outside, hit_tri_index, u, v);

            if (hit_tri_index != -1) {
                intersections[path_index].t = t_min;
                intersections[path_index].materialId = triangles[hit_tri_index].materialid;
                intersections[path_index].surfaceNormal = normal;
                intersections[path_index].outside = outside;

                // interpolate uv and store in intersection for texture sampling 
                glm::vec2 uv0 = triangles[hit_tri_index].uv[0];
                glm::vec2 uv1 = triangles[hit_tri_index].uv[1];
                glm::vec2 uv2 = triangles[hit_tri_index].uv[2];

                glm::vec2 baryUV = (1.0f - u - v) * uv0 + u * uv1 + v * uv2;

                intersections[path_index].uv = baryUV;
            }
            #else // dont forget to update this before testing performance
            for (int i = 0; i < triangles_size; i++) {
                Triangle& tri = triangles[i];

                t = triangleIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, outside, u, v);

                if (t > 0.0f && t_min > t) {
                    t_min = t;
                    hit_tri_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
            }

            if (hit_tri_index != -1) { // hit a triangle
                intersections[path_index].t = t_min;
                intersections[path_index].materialId = triangles[hit_tri_index].materialid;
                intersections[path_index].surfaceNormal = normal;
                intersections[path_index].outside = outside;
            }
            #endif
        }
    }
}


/**
* Use computed intersections to sample bsdfs.
* 
* If scene uses environment map, sample when ray misses geometry. 
*/
__global__ void shadeFakeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t envmap,
    cudaTextureObject_t* textures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) 
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;

                if (material.type == DIFFUSE || material.type == SPECULAR) {
                    scatterRayFake(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
                }
                else if (material.type == METALLICWORKFLOW) {

                    // sample texture 
                    glm::vec3 texAlbedo = glm::vec3(0.0f, 0.0f, 0.0f);
                    if (material.texIdx != -1) {
                        float4 texel = tex2D<float4>(textures[material.texIdx], intersection.uv[0], intersection.uv[1]);
                        texAlbedo = glm::vec3(texel.x, texel.y, texel.z);
                    }

                    // sample roughness map 
                    glm::vec3 texRough = glm::vec3(0.0f, 0.0f, 0.0f);
                    if (material.roughmapIdx != -1) {
                        float4 texel = tex2D<float4>(textures[material.roughmapIdx], intersection.uv[0], intersection.uv[1]);
                        texRough = glm::vec3(0.0f, texel.y, texel.z);
                    }

                    scatterRayOpaque(pathSegments[idx], intersect, intersection.surfaceNormal, texAlbedo, texRough, material, rng);
                }
                else if (material.type == DIELECTRIC) {
                    scatterRayTransparent(pathSegments[idx], intersect, intersection.surfaceNormal, material, intersection.outside, rng);
                    
                }
            }
        }
        else {
            glm::vec3 dir = pathSegments[idx].ray.direction;

            if (envmap) {
                // world -> spherical -> uv
                float theta = atan2f(dir[2], dir[0]);
                float phi = asinf(dir[1]);

                float u = (theta + PI) / (TWO_PI);
                float v = (phi + PI * 0.5f) / PI;

                // sample from envmap
                float4 radiance = tex2D<float4>(envmap, u, v);
                float r = radiance.x;
                float g = radiance.y;
                float b = radiance.z;

                pathSegments[idx].color *= glm::vec3(r, g, b);
            }
            else { // if no environment map, shade black
                pathSegments[idx].color = glm::vec3(0.0f);
            }

            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        // simple clamping to reduce bright fireflies
        glm::vec3 accColor = iterationPaths[index].color;
        accColor = glm::min(accColor, glm::vec3(10.0f));
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += accColor;
    }
}

struct is_alive {
    __host__ __device__ bool operator()(const PathSegment& ps) const {
        return ps.remainingBounces > 0;
    }
};

struct compare_by_material {
    __host__ __device__ bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2) const {
        return s1.materialId < s2.materialId;
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
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            hst_scene->triangles.size(),
            dev_intersections,
            dev_nodes,
            dev_triPtrs
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        #if MATERIAL_SORTING
            thrust::device_ptr<ShadeableIntersection> d_intersections(dev_intersections);
            thrust::device_ptr<PathSegment> d_paths_sorted(dev_paths);

            thrust::stable_sort_by_key(thrust::device, d_intersections, d_intersections + num_paths, d_paths_sorted, compare_by_material());
        #endif

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_envmap,
            dev_textures
        );

        #if STREAM_COMPACTION
            thrust::device_ptr<PathSegment> d_paths_compact(dev_paths);
            thrust::device_ptr<PathSegment> compact_end = thrust::stable_partition(thrust::device, d_paths_compact, d_paths_compact + num_paths, is_alive());
            num_paths = compact_end.get() - dev_paths;
        #endif

            
        iterationComplete = (num_paths == 0 || depth >= traceDepth); 

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // denoise accumulated dev_image

    if (USE_DENOISER && (iter % 10 == 0)) {
        denoise();
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised);
    }
    else if (USE_DENOISER) {
        int iterAfterDenoise = iter % 10;
        int denoisedIter = iter - iterAfterDenoise;
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, denoisedIter, dev_denoised);
    }
    else {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    }
   
    checkCUDAError("pathtrace");
}

void pathtraceCopyImg() {
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // Retrieve image from GPU - don't want to do every frame like base code does
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}


