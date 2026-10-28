#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "perf.h"

#define ERRORCHECK 1

// Toggle stream compaction of terminated paths (for performance comparison)
#define STREAM_COMPACTION 1

// Sort paths by material before starting shading
#define SORT_BY_MATERIAL 1

// Anti-aliasing
#define ANTIALIASING 1

// Skip a mesh's triangle loop when the ray misses its bounding box
#define MESH_AABB_CULL 1

// Thin lens camera: rays start on a disk instead of a point, so only one distance is sharp
#define DEPTH_OF_FIELD 1

// Walk each mesh's bounding volume hierarchy instead of testing all its triangles
#define BVH 1

// most nodes that can wait to be visited at once; a balanced tree over a million triangles needs about 20
#define BVH_STACK_SIZE 32

// Aim the last ray of a path at a random point on a light instead of a random direction
#define DIRECT_LIGHTING 0

// Debug view: paint the first hit with its normal mapped to [0,1] instead of shading
#define DEBUG_NORMALS 0

// Debug view: paint the first hit with its uv as (u, v, 0) instead of shading
#define DEBUG_UV 0

// Debug view: paint the first hit with its tangent mapped to [0,1] instead of shading
#define DEBUG_TANGENT 0
// Debug view: paint the first hit with its bumped normal mapped to [0,1]
#define DEBUG_BUMP 0

// Debug view: paint each path by how it ended, depth exhausted red, NaN direction green,
// NaN origin cyan, genuine miss blue
#define DEBUG_TERMINATION 0

// TEXTURE_BILINEAR lives in interactions.h, next to sampleTexture, because
// interactions.cu cannot see the defines in this file

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
static Triangle* dev_triangles = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static glm::vec3* dev_texels = NULL;
static TextureInfo* dev_textures = NULL;
static int* dev_lights = NULL;
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

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_texels, scene->texels.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_texels, scene->texels.data(), scene->texels.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(TextureInfo));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(TextureInfo), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(int));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_triangles);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_texels);
    cudaFree(dev_textures);
    cudaFree(dev_lights);
    cudaFree(dev_bvhNodes);

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

#if ANTIALIASING
        // antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + (u01(rng) - 0.5f) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + (u01(rng) - 0.5f) - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

#if DEPTH_OF_FIELD
        if (cam.lensRadius > 0.0f)
        {
            // seeded apart from the antialiasing jitter so the two do not move together
            thrust::default_random_engine lensRng = makeSeededRandomEngine(iter, index, traceDepth + 1);
            thrust::uniform_real_distribution<float> lens01(0, 1);

            // the point this pixel sees on the plane of focus
            // the plane sits focalDistance away measured along cam.view, not along the ray,
            // so a ray toward the edge of the image travels farther to reach it
            float t = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 focusPoint = segment.ray.origin + t * segment.ray.direction;

            // uniform random point on the lens; the square root spreads the samples evenly,
            // a radius taken straight from the random number crowds them in the middle
            float r = cam.lensRadius * glm::sqrt(lens01(lensRng));
            float angle = 2.0f * PI * lens01(lensRng);
            glm::vec2 lens(r * glm::cos(angle), r * glm::sin(angle));

            // start from that point on the lens and aim at the same focus point
            segment.ray.origin += cam.right * lens.x + cam.up * lens.y;
            segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);
        }
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


// Tests triangles[start, start + count) and keeps the nearest hit closer than meshT
// outputs go through locals so a farther triangle cannot overwrite a nearer hit
__device__ void intersectTriangleRange(
    const Triangle* triangles,
    int start,
    int count,
    const Ray& ray,
    float& meshT,
    glm::vec3& point,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    bool& outside)
{
    for (int j = start; j < start + count; j++)
    {
        glm::vec3 triPoint;
        glm::vec3 triNormal;
        glm::vec2 triUV;
        glm::vec3 triTangent;
        bool triOutside;
        float tt = triangleIntersectionTest(triangles[j], ray, triPoint, triNormal, triUV, triTangent, triOutside);
        if (tt > 0.0f && tt < meshT)
        {
            meshT = tt;
            point = triPoint;
            normal = triNormal;
            uv = triUV;
            tangent = triTangent;
            outside = triOutside;
        }
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
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        glm::vec3 tmp_tangent;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            // reset so a geom whose branch does not run cannot inherit the previous t
            t = -1.0f;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_uv = glm::vec2(0.0, 0.0);
                // no uvs on the sphere yet, so any direction perpendicular to the normal
                tmp_tangent = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), tmp_normal) + glm::vec3(1e-6f, 0.0f, 0.0f));
            }
            else if (geom.type == MESH)
            {
                float meshT = FLT_MAX;
#if BVH
                // no recursion on the GPU, so the nodes still to visit wait on a small stack
                int stack[BVH_STACK_SIZE];
                int stackSize = 0;
                stack[stackSize++] = geom.bvhRoot;
                while (stackSize > 0)
                {
                    const BVHNode& node = bvhNodes[stack[--stackSize]];

                    // a ray that misses the box misses everything inside it
                    if (!aabbIntersectionTest(node.aabbMin, node.aabbMax, pathSegment.ray)) continue;

                    if (node.triCount > 0)
                    {
                        // leaf
                        intersectTriangleRange(triangles, node.triStart, node.triCount, pathSegment.ray,
                            meshT, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);
                    }
                    else if (stackSize + 2 <= BVH_STACK_SIZE)
                    {
                        // interior, both children wait their turn
                        stack[stackSize++] = node.left;
                        stack[stackSize++] = node.right;
                    }
                }
#else
#if MESH_AABB_CULL
                // ray misses the bounding box, skip the whole triangle loop (t stays -1)
                if (!aabbIntersectionTest(geom.aabbMin, geom.aabbMax, pathSegment.ray)) continue;
#endif
                intersectTriangleRange(triangles, geom.triStart, geom.triCount, pathSegment.ray,
                    meshT, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);
#endif
                if (meshT < FLT_MAX) t = meshT;
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                tangent = tmp_tangent;
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
            intersections[path_index].uv = uv;
            intersections[path_index].tangent = tangent;
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

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* texels,
    TextureInfo* textures,
    Geom* geoms,
    int* lights,
    int num_lights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces <= 0) return;
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
#if DEBUG_NORMALS
            pathSegments[idx].color = intersection.surfaceNormal * 0.5f + 0.5f;
            pathSegments[idx].remainingBounces = 0;
            return;
#endif
#if DEBUG_UV
            pathSegments[idx].color = glm::vec3(intersection.uv, 0.0f);
            pathSegments[idx].remainingBounces = 0;
            return;
#endif
#if DEBUG_TANGENT
            pathSegments[idx].color = intersection.tangent * 0.5f + 0.5f;
            pathSegments[idx].remainingBounces = 0;
            return;
#endif
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            // material is a local copy, so a textured hit can overwrite its color
            // and scatterRay picks it up unchanged
            if (material.surface.image >= 0) {
                material.color = sampleTexture(texels, textures[material.surface.image], intersection.uv);
            }
            else if (material.surface.procedural == PROC_TILES) {
                material.color = proceduralTiles(material, intersection.uv);
            }
            // keep the real surface normal; bump only changes the one used for shading
            glm::vec3 geomNormal = intersection.surfaceNormal;
            if (material.surface.bump > 0.0f && material.surface.procedural == PROC_TILES) {
                intersection.surfaceNormal = bumpNormal(intersection.surfaceNormal, intersection.tangent, material, intersection.uv);
            }
#if DEBUG_BUMP
            pathSegments[idx].color = intersection.surfaceNormal * 0.5f + 0.5f;
            pathSegments[idx].remainingBounces = 0;
            return;
#endif
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }

            // Otherwise, do BSDF lighting computation
            else {
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
                bool aimedAtLight = false;

#if DIRECT_LIGHTING
                // a path is cut off when remainingBounces reaches 0, and it is decremented
                // below, so the ray made here is the last one traced when 2 remain now
                bool lastRay = pathSegments[idx].remainingBounces == 2;
                bool diffuse = material.type == DIFFUSE;
                if (lastRay && diffuse && num_lights > 0)
                {
                    aimedAtLight = true;

                    // one light uniformly, then one point on it uniformly by area
                    int pick = glm::min((int)(u01(rng) * num_lights), num_lights - 1);
                    glm::vec3 lightPoint, lightNormal;
                    float lightArea;
                    sampleBoxLight(geoms[lights[pick]], rng, lightPoint, lightNormal, lightArea);

                    // unit direction from the hit to the light point, and the distance
                    glm::vec3 toLight = lightPoint - intersect;
                    float dist = glm::length(toLight);
                    toLight = glm::normalize(toLight);

                    // the dot product of two unit vectors is the cosine of their angle
                    // how squarely the surface faces the light, and the light faces the surface
                    float cosSurface = glm::dot(toLight, intersection.surfaceNormal);
                    float cosLight = glm::dot(-toLight, lightNormal);

                    if (cosSurface <= 0.0f || cosLight <= 0.0f || glm::dot(toLight, geomNormal) <= 0.0f)
                    {
                        // the sampled point faces away, or sits behind the surface
                        pathSegments[idx].color = glm::vec3(0.0f);
                    }
                    else
                    {
                        // the weight of aiming at this point instead of a random direction
                        //     albedo / PI                   the diffuse BSDF
                        //     cosSurface                    light arriving at a slant
                        //     cosLight / dist^2             how small the light patch looks from here
                        //     lightArea * num_lights        one over the probability of this point
                        float weight = cosSurface * cosLight / (dist * dist) * lightArea * (float)num_lights / PI;
                        pathSegments[idx].color *= material.color * weight;
                    }
                    pathSegments[idx].ray.direction = toLight;
                    pathSegments[idx].ray.origin = intersect + 0.001f * toLight;
                }
#endif

                if (!aimedAtLight)
                {
                    scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, geomNormal, material, rng);
                }
                pathSegments[idx].remainingBounces--;

#if DEBUG_TERMINATION
                if (pathSegments[idx].remainingBounces == 0) pathSegments[idx].color = glm::vec3(1.0f, 0.0f, 0.0f);
#else
                if (pathSegments[idx].remainingBounces == 0) pathSegments[idx].color = glm::vec3(0.0f);
#endif
            }

        }
        // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        else {
#if DEBUG_TERMINATION
            {
                glm::vec3 d = pathSegments[idx].ray.direction, o = pathSegments[idx].ray.origin;
                if (isnan(d.x) || isnan(d.y) || isnan(d.z)) pathSegments[idx].color = glm::vec3(0.0f, 1.0f, 0.0f);
                else if (isnan(o.x) || isnan(o.y) || isnan(o.z)) pathSegments[idx].color = glm::vec3(0.0f, 1.0f, 1.0f);
                else pathSegments[idx].color = glm::vec3(0.0f, 0.0f, 1.0f);
            }
#else
            pathSegments[idx].color = glm::vec3(0.0f);
#endif
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
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct is_path_alive
{
    __host__ __device__
        const bool operator()(const PathSegment& p)
    {
        return p.remainingBounces > 0;
    }
};

struct material_id_less
{
    __host__ __device__
        const bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
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

    perfBegin();
    perfMark(PERF_GENERATE);

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
        perfMark(PERF_INTERSECT);
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_bvhNodes,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;


        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.

#if SORT_BY_MATERIAL
        perfMark(PERF_SORT);
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_id_less());
#endif

        perfMark(PERF_SHADE);
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_texels,
            dev_textures,
            dev_geoms,
            dev_lights,
            (int)hst_scene->lights.size()
        );

#if STREAM_COMPACTION
        perfMark(PERF_COMPACT);
        PathSegment* streamCompactPartition = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_path_alive());
        num_paths = streamCompactPartition - dev_paths;
#endif

        perfPathsAlive(iter, depth, num_paths);

        iterationComplete = (depth == traceDepth) || (num_paths == 0);


        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    perfMark(PERF_GATHER);
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    perfMark(PERF_STAGES);
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // the cudaMemcpy above waited for the GPU, so every stage has finished
    perfEnd(iter);

    checkCUDAError("pathtrace");
}
