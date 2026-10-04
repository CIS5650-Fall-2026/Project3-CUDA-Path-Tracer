#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/remove.h>

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




__device__ glm::vec3 toneMap(glm::vec3 color)
{
  
//    color *= 1.5f;
//    // Reinhard tone mapping
      //color = color / (color + glm::vec3(1.0f));
//
//    // Gamma correction
      //color = glm::pow(color, glm::vec3(1.0f / 1.4f));

    return color;
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

        // tonemap step
        pix = toneMap( pix / (float)iter );

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

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
static BVHNode* dev_bvhnodes = NULL;
static Triangle* dev_triangles = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_pathMaterialIds = NULL;
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

    cudaMalloc(&dev_pathMaterialIds, pixelcount * sizeof(int));
    cudaMemset(dev_pathMaterialIds, 0, pixelcount * sizeof(int));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvhnodes, scene->bvhNodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvhnodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
	cudaFree(dev_bvhnodes);
	cudaFree(dev_triangles);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_pathMaterialIds);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

__device__ __host__ glm::vec2 SampleUniformDiskConcentric(glm::vec2 u)
{
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f);

    if (uOffset.x == 0.0f && uOffset.y == 0.0f)
        return glm::vec2(0.0f);

    float theta;
    float r;

    if (fabs(uOffset.x) > fabs(uOffset.y))
    {
        r = uOffset.x;
        theta = (PI / 4.0f) * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta =
            (PI / 2.0f) -
            (PI / 4.0f) * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(cosf(theta), sinf(theta));
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

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        // antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
        thrust::uniform_real_distribution<float> u01(0, 1);
        segment.ray.direction.x += (u01(rng) - 0.5f) * cam.pixelLength.x;
        segment.ray.direction.y += (u01(rng) - 0.5f) * cam.pixelLength.y;

        // calculate DOF ray
		// 1 - intersection point on focal plane
		float focalDistance = cam.focalDistance;
		float t = focalDistance / glm::dot(segment.ray.direction, cam.view);
		glm::vec3 focalPoint = segment.ray.origin + segment.ray.direction * t;
		// 2 - jitter ray origin on lens
		glm::vec2 offset = SampleUniformDiskConcentric(glm::vec2(u01(rng), u01(rng))) * cam.lensRadius;
		segment.ray.origin += cam.right * offset.x + cam.up * offset.y;
		segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

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
    BVHNode* bvhNodes,
    Triangle* triangles,
	int bvhRootIndex,
    Geom* geoms,
    int* dev_pathMaterialIds,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {

        PathSegment pathSegment = pathSegments[path_index];

        // Branch here based on if we have BVH on or NOT
        if (USE_BVH) {
			// traverse the BVH to find the intersection
			float closestT = FLT_MAX;
			glm::vec3 hitPoint;
			glm::vec3 hitNormal;
			int materialId = -1;
			bool hit = BVHIntersectionTest(bvhRootIndex, pathSegment.ray, bvhNodes, triangles, geoms, closestT, hitPoint, hitNormal, materialId);

            if (!hit)
            {

                intersections[path_index].t = -1.0f;
                dev_pathMaterialIds[path_index] = -1;

            }
            else
            {

                intersections[path_index].t = closestT;
                intersections[path_index].materialId = materialId;
                intersections[path_index].surfaceNormal = hitNormal;

                dev_pathMaterialIds[path_index] = materialId;

            }

        }
        if (!USE_BVH) {

            int hit_geom_index = -1;

            float t;
            glm::vec3 intersect_point;
            glm::vec3 normal;
            float t_min = FLT_MAX;

            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;

            // naive intersection test
            for (int i = 0; i < geoms_size; i++)
            {
                Geom& geom = geoms[i];

				t = anyGeomIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);

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

                dev_pathMaterialIds[path_index] = -1;

            }
            else
            {
                // The ray hits something
                intersections[path_index].t = t_min;
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                intersections[path_index].surfaceNormal = normal;

                dev_pathMaterialIds[path_index] = geoms[hit_geom_index].materialid;

            }
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
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
		PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            // If intersection exists:
			int bouncesLeft = pathSegment.remainingBounces;
            // if no bounces left: don't color and return
            if (bouncesLeft <= 0) {
                return;
			}

            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, bouncesLeft);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Read variables
            // material: the material of the intersection
            // materialColor: base color of the material
            // intersectionPos: intersection position (already offset by e)
            // nor: surface normal of intersection
			// wo 
			// ray
			Ray& ray = pathSegment.ray;
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
			glm::vec3 intersectionPos = getPointOnRay(ray, intersection.t);
			glm::vec3 nor = intersection.surfaceNormal;
            glm::vec3 wo = ray.direction;

            switch (material.type)
            {
            case EMISSIVE:

                pathSegment.color *= materialColor * material.emittance;
                pathSegment.remainingBounces = -1;
                break;

            case DIFFUSE:

                // Scatter Ray
                scatterRay(pathSegment, intersectionPos, nor, material, rng);
                pathSegment.remainingBounces--;

                // Shade
                {
                    float lambertTerm = glm::dot(nor, ray.direction);
                    glm::vec3 bsdf = materialColor / PI;
                    float pdf = lambertTerm / PI;
                    pathSegment.color *= bsdf * lambertTerm / pdf;
                }

                break;

            case REFLECTIVE:

                // Reflect Ray
                ray.direction = glm::reflect(ray.direction, nor);
                ray.origin = intersectionPos;
                pathSegment.remainingBounces--;

                // Shade
                pathSegment.color *= materialColor;

                break;

            case REFRACTIVE:
                {
                    glm::vec3 n = nor;                     
                    float ior = material.indexOfRefraction;
                    glm::vec3 dirIn = glm::normalize(ray.direction);

                    float eta;
                    if (glm::dot(dirIn, n) < 0.0f) {
                        eta = 1.0f / ior;                  
                    }
                    else {
                        eta = ior;                        
                        n = -n;                           
                    }

                    glm::vec3 newDir = glm::refract(dirIn, n, eta);

                    if (glm::dot(newDir, newDir) < 1e-8f) {
                        // total internal reflection
                        newDir = glm::reflect(dirIn, n);
                        ray.origin = intersectionPos + n * 0.001f; 
                    }
                    else {
                        ray.origin = intersectionPos - n * 0.001f; 
                    }

                    ray.direction = glm::normalize(newDir);
                    pathSegment.color *= materialColor;
                    pathSegment.remainingBounces--;
                }
                break;

            case DIELECTRIC:

                {
                    float cosTheta = glm::dot(-ray.direction, nor);

                    float etaI = 1.0f;
                    float etaT = material.indexOfRefraction;

					bool refract = false;

                    if (cosTheta < 0.0f)
                    {
                        // Exiting the material
                        cosTheta = -cosTheta;
						float temp = etaI;
						etaI = etaT;
						etaT = temp;
                        nor = -nor;

                        refract = true;
                    }

                    float F = fresnelDielectric(cosTheta, etaI, etaT);


                    float eta = etaI / etaT;
                    // Check for total internal reflection
                    float sinThetaT2 = eta * eta * (1.0f - cosTheta * cosTheta);

                    // Randomly pick between reflection and refraction based on Fresnel term
                    float randomTerm = u01(rng);
                    float randomThreshold = F;

					if (randomTerm < F && !refract) { // Reflect

                        // Reflect Ray
                        ray.direction = glm::reflect(ray.direction, nor);
                        ray.origin = intersectionPos;
                        pathSegment.remainingBounces--;

                        // Shade
                        pathSegment.color *= materialColor;


                    }
					else { // Refract

                        // make ray pass through the surface
                        refractRay(ray.direction, nor, eta);
                        ray.origin = intersectionPos + ray.direction * 0.0002f;
                        pathSegment.remainingBounces--;

                        // Shade
                        pathSegment.color *= materialColor;

                    }

                }
                break;

            case MICROFACET:
                // TODO
                break;
            }

        }
        else {
            // If no intersection: Environment (Currently: Black) TODO: Env. Map
            pathSegment.color *= BACKGROUND_COLOR;
            pathSegment.remainingBounces = -1;
        }

        if (pathSegment.remainingBounces == 0) {
			pathSegment.color = glm::vec3(0.0f);
			pathSegment.remainingBounces = -1;
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
	int maxDepth = hst_scene->state.traceDepth;

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
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_bvhnodes,
            dev_triangles,
			hst_scene->bvhRootNodeIndex,
            dev_geoms,
            dev_pathMaterialIds,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // Optional: Shuffle pathsegemnts based on material
        bool shufflePathSegments = false;
        if (shufflePathSegments) {

            thrust::device_ptr<int> dev_thrust_materialIds(dev_pathMaterialIds);
            thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
            thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);

            thrust::device_vector<int> materialIdsCopy(dev_thrust_materialIds, dev_thrust_materialIds + num_paths);

            thrust::sort_by_key(
                dev_thrust_materialIds,
                dev_thrust_materialIds + num_paths,
                dev_thrust_paths
            );
            thrust::sort_by_key(
                materialIdsCopy.begin(),
                materialIdsCopy.end(),
                dev_thrust_intersections
            );

        }

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        if (depth > maxDepth) {
            iterationComplete = true;
        }

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
