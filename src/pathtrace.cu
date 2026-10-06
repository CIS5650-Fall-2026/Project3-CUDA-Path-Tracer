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
#include "bvh.h"

#define ERRORCHECK 1

// can toggle each feature for performance comparison
#define CONTIGUOUS_BY_MATERIAL 1
#define DIRECT_LIGHTING 1
#define MIS 1
#define RUSSIAN_ROULETTE 1
#define HIERARCHICAL_SPATIAL_DS 1
#define TEXTURE_MAPPING 1
#define BUMP_MAPPING 1
#define ENVIRONMENT_LIGHT 0
#define HALTON_SEQUENCE 0

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

// predicate for thrust partition
struct hasBouncesRemaining {
    __host__ __device__ bool operator()(const PathSegment &path) const {
        return path.remainingBounces > 0;
    }
};

struct compareMaterial {
    __host__ __device__ bool operator()(
        const ShadeableIntersection &a,
        const ShadeableIntersection &b) const {
            return a.materialId < b.materialId;
        }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__host__ __device__ float halton(unsigned int index, unsigned int base) {
    // reverse digits and add decimal point
    float value = 0.0f;
    float inverseBase = 1.0f / base;
    while (index > 0) {
        value += (index % base) * inverseBase;
        index /= base;
        inverseBase /= base;
    }

    return value;
}

// helpers for direct lighting
__host__ __device__
float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    float f2 = f * f;
    float g2 = g * g;

    return f2 + g2 > 0.0f ? f2 / (f2 + g2) : 0.0f;
}

__device__
float TriangleArea(const Geom& triangle) {
    glm::vec3 a = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    glm::vec3 b = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    return glm::length(glm::cross(a, b)) / 2.0f;
}

__device__
glm::vec3 TriangleNormal(const Geom& triangle)
{
    glm::vec3 a = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    glm::vec3 b = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    return glm::normalize(glm::cross(a, b));
}

__device__
float IntersectGeom(const Geom& geom, const Ray& ray, glm::vec3& intersectionPoint, glm::vec3& normal){
    bool outside = true;
    if (geom.type == CUBE) {
        return boxIntersectionTest(
            geom,
            ray,
            intersectionPoint,
            normal,
            outside
        );
    }
    if (geom.type == SPHERE) {
        return sphereIntersectionTest(
            geom,
            ray,
            intersectionPoint,
            normal,
            outside
        );
    }
    if (geom.type == TRIANGLE) {
        return triangleIntersectionTest(
            geom,
            ray,
            intersectionPoint,
            normal,
            outside
        );
    }

    return -1.0f;
}

__device__
bool IntersectBounds(
    const Bounds3& bounds,
    const Ray& ray,
    float tMax,
    const glm::vec3& invDir,
    const int dirIsNeg[3])
{
    float t0 = 0.0f;
    float t1 = tMax;

    for (int axis = 0; axis < 3; ++axis) {
        if (ray.direction[axis] == 0.0f) {
            if (ray.origin[axis] < bounds.pMin[axis] || ray.origin[axis] > bounds.pMax[axis]) {
                return false;
            }
            continue;
        }

        float nearBound = dirIsNeg[axis] ? bounds.pMax[axis] : bounds.pMin[axis];
        float farBound = dirIsNeg[axis] ? bounds.pMin[axis] : bounds.pMax[axis];

        float tNear = (nearBound - ray.origin[axis]) * invDir[axis];

        float tFar = (farBound - ray.origin[axis]) * invDir[axis];

        t0 = glm::max(t0, tNear);
        t1 = glm::min(t1, tFar);

        if (t0 > t1) {
            return false;
        }
    }

    return true;
}

__device__
void IntersectBVH(
    const Ray& ray,
    Geom* geoms,
    LinearBVHNode* nodes,
    int* orderedGeomIndices,
    float& tMin,
    int& hitGeomIndex,
    glm::vec3& intersectionPoint,
    glm::vec3& normal)
{
    glm::vec3 invDir = 1.0f / ray.direction;
    int dirIsNeg[3] = {
        invDir.x < 0,
        invDir.y < 0,
        invDir.z < 0
    };

    int nodesToVisit[64];
    int toVisitOffset = 0;
    int currentNodeIndex = 0;

    while (true) {
        const LinearBVHNode& node = nodes[currentNodeIndex];

        if (IntersectBounds(node.bounds, ray, tMin, invDir, dirIsNeg)) {
            if (node.nPrimitives > 0) {
                for (int i = 0; i < node.nPrimitives; ++i) {
                    int geomIndex = orderedGeomIndices[node.primitivesOffset + i];
                    glm::vec3 candidatePoint;
                    glm::vec3 candidateNormal;
                    float t = IntersectGeom(geoms[geomIndex], ray, candidatePoint, candidateNormal);
                    if (t > 0 && t < tMin) {
                        tMin = t;
                        hitGeomIndex = geomIndex;
                        intersectionPoint = candidatePoint;
                        normal = candidateNormal;
                    }
                }

                if (toVisitOffset == 0) {
                    break;
                }
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                if (dirIsNeg[node.axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node.secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node.secondChildOffset;
                    currentNodeIndex++;
                }
            }
        } else {
            if (toVisitOffset == 0) {
                break;
            }
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
}

__device__
bool IntersectP(
    const Ray& ray,
    float tMax,
    Geom* geoms,
    int geoms_size,
    LinearBVHNode *nodes,
    int *orderedGeomIndices,
    int nodeCount
) {
    #if HIERARCHICAL_SPATIAL_DS
    if (nodeCount == 0) {
        return false;
    }
    if (tMax <= 0) {
        return false;
    }

    glm::vec3 invDir = 1.0f / ray.direction;
    int dirIsNeg[3] = {
        invDir.x < 0,
        invDir.y < 0,
        invDir.z < 0
    };
    
    int nodeToVisit[64];
    int toVisitOffset = 0;
    int currNode = 0;

    while (true) {
        const LinearBVHNode &node = nodes[currNode];
        if (IntersectBounds(node.bounds, ray, tMax, invDir, dirIsNeg)) {
            if (node.nPrimitives > 0) {
                for (int i = 0; i < node.nPrimitives; ++i) {
                    int geomIndex = orderedGeomIndices[node.primitivesOffset + i];
                    glm::vec3 intersectionPoint;
                    glm::vec3 normal;
                    float t = IntersectGeom(geoms[geomIndex], ray, intersectionPoint, normal);
                    if (t > 0 && t < tMax) {
                        return true;
                    }
                }

                if (toVisitOffset == 0) {
                    break;
                }
                currNode = nodeToVisit[--toVisitOffset];
            } else if (dirIsNeg[node.axis]) {
                nodeToVisit[toVisitOffset++] = currNode + 1;
                currNode = node.secondChildOffset;
            } else {
                nodeToVisit[toVisitOffset++] = node.secondChildOffset;
                currNode++;
            }
        } else {
            if (toVisitOffset == 0) {
                break;
            }
            currNode = nodeToVisit[--toVisitOffset];
        }
    }
    return false;
    #else
    for (int i = 0; i < geoms_size; ++i) {
        glm::vec3 intersectionPoint;
        glm::vec3 normal;
        bool outside = true;
        float t = -1;

        if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
        } else if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
        } else if (geoms[i].type == TRIANGLE) {
            t = triangleIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
        }

        if (t > 0 && t < tMax - 0.001) {
            return true;
        }
    }

    return false;
    #endif
}

__device__
float LightPDF(const Geom& light, int numLights, const Ray& ray, float t) {
    if (numLights == 0) {
        return 0;
    }

    float area = TriangleArea(light);
    float cosLight = glm::max(glm::dot(TriangleNormal(light), -ray.direction), 0.0f);

    if (area <= EPSILON || cosLight <= 0.0f) {
        return 0;
    }

    float lightPMF = 1.0f / numLights;
    return lightPMF * t * t / (area * cosLight);
}

__device__
glm::vec3 SampleLd(
    const glm::vec3& p, const glm::vec3& n, const Material& material,
    Geom* geoms, int geoms_size, Material* materials, int* lightGeomIndices,
    int numLights, thrust::default_random_engine& rng,
    LinearBVHNode* bvhNodes, int* bvhGeomIndices, int bvhNodeCount, int iter
) {
    if (numLights == 0) {
        return glm::vec3(0);
    }

    #if HALTON_SEQUENCE
    int lightIndex = (int)(halton(iter, 17) * numLights);
    #else
    thrust::uniform_real_distribution<float> u01(0, 1);

    int lightIndex = (int)(u01(rng) * numLights);
    #endif
    if (lightIndex == numLights) {
        lightIndex--;
    }

    const Geom& light = geoms[lightGeomIndices[lightIndex]];
    float area = TriangleArea(light);

    if (area <= EPSILON) {
        return glm::vec3(0.0f);
    }

    #if HALTON_SEQUENCE
    float sqrtU = sqrtf(halton(iter, 11));
    float b0 = 1.0f - sqrtU;
    float b1 = halton(iter, 19) * sqrtU;
    float b2 = 1.0f - b0 - b1;
    #else
    float sqrtU = sqrtf(u01(rng));
    float b0 = 1.0f - sqrtU;
    float b1 = u01(rng) * sqrtU;
    float b2 = 1.0f - b0 - b1;
    #endif

    glm::vec3 pLight =
        b0 * light.triangleVertices[0] +
        b1 * light.triangleVertices[1] +
        b2 * light.triangleVertices[2];

    Ray shadowRay;
    shadowRay.origin = p + 0.001f * n;

    glm::vec3 toLight = pLight - shadowRay.origin;
    float distanceSquared = glm::dot(toLight, toLight);

    if (distanceSquared <= EPSILON) {
        return glm::vec3(0);
    }

    float distance = sqrtf(distanceSquared);
    shadowRay.direction = toLight / distance;

    float cosTheta = glm::max(glm::dot(n, shadowRay.direction), 0.0f);
    float cosLight = glm::max(glm::dot(TriangleNormal(light), -shadowRay.direction), 0.0f);

    if (cosTheta <= 0 || cosLight <= 0 || 
        IntersectP(shadowRay, distance, geoms, geoms_size, bvhNodes, bvhGeomIndices, bvhNodeCount)
    ) {
        return glm::vec3(0);
    }

    float lightPMF = 1.0f / numLights;
    float p_l = lightPMF * distanceSquared / (area * cosLight);
    float p_b = cosTheta / PI;
    float w_l = 1;

#if MIS
    w_l = PowerHeuristic(1, p_l, 1, p_b);
#endif

    glm::vec3 f = material.color / PI;
    const Material& lightMaterial = materials[light.materialid];
    glm::vec3 Li = lightMaterial.color * lightMaterial.emittance;

    return w_l * Li * f * cosTheta / p_l;
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
// static variables for device memory, any extra info you need, etc
static int *dev_light_geom_indices = NULL;
static int num_lights = 0;
static LinearBVHNode* dev_bvh_nodes = NULL;
static int* dev_bvh_geom_indices = NULL;
static int num_bvh_nodes = 0;
static uchar4 *dev_tex_pixels = NULL;

static std::vector<LinearBVHNode> hst_bvh_nodes;
static std::vector<int> hst_bvh_geom_indices;

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

    #if HIERARCHICAL_SPATIAL_DS
    if (hst_bvh_nodes.empty()) {
        BuildBVH(scene->geoms, 4, hst_bvh_nodes, hst_bvh_geom_indices);
    }

    num_bvh_nodes = hst_bvh_nodes.size();
    if (num_bvh_nodes > 0) {
        cudaMalloc(&dev_bvh_nodes, num_bvh_nodes * sizeof(LinearBVHNode));
        cudaMemcpy(dev_bvh_nodes, hst_bvh_nodes.data(), num_bvh_nodes * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_bvh_geom_indices, hst_bvh_geom_indices.size() * sizeof(int));
        cudaMemcpy(dev_bvh_geom_indices, hst_bvh_geom_indices.data(), hst_bvh_geom_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    #endif


    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // initialize any extra device memeory you need
    // initializing all sources of light for direct lighting and MIS

    std::vector<int> lightGeomIndices;
    for (int i = 0; i < scene->geoms.size(); ++i) {
        const Geom &curr_geom = scene->geoms[i];

        if (curr_geom.type != TRIANGLE || curr_geom.materialid < 0 || curr_geom.materialid >= scene->materials.size()) {
            continue;
        }

        if (scene->materials[curr_geom.materialid].emittance > 0) {
            lightGeomIndices.push_back(i);
        }
    }

    num_lights = lightGeomIndices.size();
    if (num_lights > 0) {
        cudaMalloc(&dev_light_geom_indices, num_lights * sizeof(int));
        cudaMemcpy(dev_light_geom_indices, lightGeomIndices.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
    }

    if (!scene->texPixels.empty()) {
        cudaMalloc(&dev_tex_pixels, scene->texPixels.size() * sizeof(uchar4));
        cudaMemcpy(dev_tex_pixels, scene->texPixels.data(), scene->texPixels.size() * sizeof(uchar4), cudaMemcpyHostToDevice);
    }


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_light_geom_indices);
    cudaFree(dev_bvh_geom_indices);
    cudaFree(dev_bvh_nodes);
    cudaFree(dev_tex_pixels);

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
        segment.color = glm::vec3(1, 1, 1);
        segment.L = glm::vec3(0);
        segment.specularBounce = false;
        segment.p_b = 0;
        segment.n = 1;

        // implement antialiasing by jittering the ray
        #if HALTON_SEQUENCE
        float jitter_x = halton(iter, 2) - 0.5;
        float jitter_y = halton(iter, 7) - 0.5;
        #else
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitter_x = u01(rng) - 0.5;
        float jitter_y = u01(rng) - 0.5;
        #endif

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitter_x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitter_y - (float)cam.resolution.y * 0.5f)
        );

        if (cam.lensRadius > 0) {
            #if HALTON_SEQUENCE
            float radius = cam.lensRadius * sqrtf(halton(iter, 3));
            float theta = 2 * PI * halton(iter, 5);
            #else
            float radius = cam.lensRadius * sqrtf(u01(rng));
            float theta = 2 * PI * u01(rng);
            #endif
            glm::vec2 pLens(radius * cosf(theta), radius * sinf(theta));
            
            float ft = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 pFocus = cam.position + ft * segment.ray.direction;

            glm::vec3 lensOffset = pLens.x * cam.right + pLens.y * cam.up;
            segment.ray.origin = cam.position + lensOffset;
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        }

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
    ShadeableIntersection* intersections,
    LinearBVHNode* bvhNodes,
    int* bvhGeomIndices,
    int bvhNodeCount)
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

        #if HIERARCHICAL_SPATIAL_DS
        if (bvhNodeCount > 0) {
            IntersectBVH(
                pathSegment.ray,
                geoms,
                bvhNodes,
                bvhGeomIndices,
                t_min,
                hit_geom_index,
                intersect_point,
                normal
            );
        }
        #else
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
            else if (geom.type == TRIANGLE) {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
        #endif

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
            #if !HALTON_SEQUENCE
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);
            #endif

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
                #if HALTON_SEQUENCE
                pathSegments[idx].color *= halton(iter, 23);
                #else
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
                #endif
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


// helpers for texture mapping
__device__ int wrapTexCoord(int x, int size) {
    x %= size;
    return x >= 0 ? x : x + size;
}

__device__ glm::vec3 readTexPixel(
    const uchar4 *pixels, int offset, int width, int height, int x, int y) {
        x = wrapTexCoord(x, width);
        y = wrapTexCoord(y, height);
        uchar4 pixel = pixels[offset + y * width + x];
        return glm::vec3(pixel.x, pixel.y, pixel.z) / 255.0f;
}

__device__ glm::vec3 sampleTexture(
    const uchar4 *pixels, int offset, int width, int height, glm::vec2 st
) {
    st.x -= floorf(st.x);
    st.y -= floorf(st.y);
    float x = st.x * width - 0.5;
    float y = st.y * height - 0.5;
    int x0 = floor(x);
    int y0 = floor(y);
    float dx = x - x0;
    float dy = y - y0;

    glm::vec3 c00 = readTexPixel(pixels, offset, width, height, x0, y0);
    glm::vec3 c10 = readTexPixel(pixels, offset, width, height, x0 + 1, y0);
    glm::vec3 c01 = readTexPixel(pixels, offset, width, height, x0, y0 + 1);
    glm::vec3 c11 = readTexPixel(pixels, offset, width, height, x0 + 1, y0 + 1);

    glm::vec3 c0 = c00 * (1 - dx) + c10 * dx;
    glm::vec3 c1 = c01 * (1 - dx) + c11 * dx;
    return c0 * (1 - dy) + c1 * dy;
}

__device__ glm::vec2 getTriangleUV(const Geom& triangle, const glm::vec3& p) {
    glm::vec3 a = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    glm::vec3 b = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    glm::vec3 pointVec = p - triangle.triangleVertices[0];

    float d00 = glm::dot(a, a);
    float d01 = glm::dot(a, b);
    float d11 = glm::dot(b, b);
    float d20 = glm::dot(pointVec, a);
    float d21 = glm::dot(pointVec, b);
    float denom = d00 * d11 - d01 * d01;
    float b1 = (d11 * d20 - d01 * d21) / denom;
    float b2 = (d00 * d21 - d01 * d20) / denom;
    float b0 = 1.0f - b1 - b2;

    return b0 * triangle.triangleUVs[0] + b1 * triangle.triangleUVs[1] + b2 * triangle.triangleUVs[2];
}

__device__ float sampleHeight(const uchar4* pixels, const Material& material, glm::vec2 st) {
    glm::vec3 color = sampleTexture(
        pixels, material.bumpTexOffset, material.bumpTexWidth,
        material.bumpTexHeight, st);

    return (color.x + color.y + color.z) / 3;
}

__device__ glm::vec3 bumpNormal(
    const uchar4* pixels, const Material& material, const Geom& triangle, glm::vec2 st, glm::vec3 normal
) {
    glm::vec3 e1 = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    glm::vec3 e2 = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    glm::vec2 dst1 = triangle.triangleUVs[1] - triangle.triangleUVs[0];
    glm::vec2 dst2 = triangle.triangleUVs[2] - triangle.triangleUVs[0];

    float determinant = dst1.x * dst2.y - dst1.y * dst2.x;
    if (fabsf(determinant) < 0.0000001) {
        return normal;
    }

    glm::vec3 dpdu = (dst2.y * e1 - dst1.y * e2) / determinant;
    glm::vec3 dpdv = (-dst2.x * e1 + dst1.x * e2) / determinant;
    float du = 1.0f / material.bumpTexWidth;
    float dv = 1.0f / material.bumpTexHeight;
    float height = sampleHeight(pixels, material, st);
    float dHdu = material.bumpStrength * (sampleHeight(pixels, material, st + glm::vec2(du, 0)) - height) / du;
    float dHdv = material.bumpStrength * (sampleHeight(pixels, material, st + glm::vec2(0, dv)) - height) / dv;

    glm::vec3 bumpedNormal = glm::normalize(glm::cross(dpdu + dHdu * normal, dpdv + dHdv * normal));
    if (glm::dot(bumpedNormal, normal) < 0) {
        bumpedNormal = -bumpedNormal;
    }

    return bumpedNormal;
}

__host__ __device__ glm::vec3 evaluateProceduralTexture(const Material& material, const glm::vec3& p) {
    if (material.textureType == CHECKER) {
        int checker = floor(2 * p.x) + floor(2 * p.y) + floor(2 * p.z);
        return checker % 2 == 0 ? material.color : 0.1f * material.color;
    } else if (material.textureType == STRIPES) {
        return sinf(12.0f * p.y) > 0.0f ? material.color : 0.1f * material.color;
    }

    return material.color;
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    int depth,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    uchar4 *texPixels,
    int* lightGeomIndices,
    int numLights,
    LinearBVHNode* bvhNodes,
    int* bvhGeomIndices,
    int bvhNodeCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment &path = pathSegments[idx];
        if (path.remainingBounces <= 0) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            glm::vec3& L = path.L;
            glm::vec3& beta = path.color;

          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth);

            Material material = materials[intersection.materialId];
            glm::vec3 p = path.ray.origin + intersection.t * path.ray.direction;

            const Geom &currGeom = geoms[intersection.geomId];
            bool hasUVs = currGeom.type == TRIANGLE && currGeom.hasUVs;
            glm::vec2 st = hasUVs ? getTriangleUV(currGeom, p) : glm::vec2(0);

            if (hasUVs && material.colorTexWidth > 0) {
                #if TEXTURE_MAPPING
                material.color *= sampleTexture(
                    texPixels, material.colorTexOffset, material.colorTexWidth, 
                    material.colorTexHeight, st);
                #else
                material.textureType = CHECKER;
                material.color = evaluateProceduralTexture(material, p);
                #endif
            } else {
                material.color = evaluateProceduralTexture(material, p);
            }

            #if BUMP_MAPPING
            if (hasUVs && material.bumpTexWidth > 0) {
                intersection.surfaceNormal = bumpNormal(
                    texPixels, material, currGeom, st, intersection.surfaceNormal);
            }
            #endif
            
            glm::vec3 materialColor = material.color;
            
            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                const Geom& light = geoms[intersection.geomId];
                bool frontFacing = light.type != TRIANGLE ||
                    glm::dot(TriangleNormal(light), -path.ray.direction) > 0;

                if (frontFacing) {
                    glm::vec3 Le = materialColor * material.emittance;

                    #if DIRECT_LIGHTING
                    if (depth == 1 || path.specularBounce || light.type != TRIANGLE) {
                        L += beta * Le;
                    }
                    
                    #if MIS
                    else {
                        float p_l = LightPDF(light, numLights, path.ray, intersection.t);
                        float w_b = PowerHeuristic(1, path.p_b, 1, p_l);
                        L += beta * w_b * Le;
                    }
                    #endif
                    
                    #else
                    L += beta * Le;
                    #endif
                }

                path.remainingBounces = 0;
            }
            else {
                #if DIRECT_LIGHTING
                if (material.hasRefractive == 0) {
                    glm::vec3 Ld = SampleLd(
                        p, intersection.surfaceNormal, material, geoms,
                        geoms_size, materials, lightGeomIndices, numLights, rng,
                        bvhNodes, bvhGeomIndices, bvhNodeCount, iter
                    );

                    L += beta * Ld;
                }
                #endif

                scatterRay(path, p, intersection.surfaceNormal, material, rng);
                path.remainingBounces--;

                if (material.hasRefractive == 0) {
                    float cosTheta = glm::max(
                    glm::dot(intersection.surfaceNormal, path.ray.direction), 0.0f);
                    path.p_b = cosTheta / PI;
                    path.specularBounce = false;
                } else {
                    path.p_b = 0;
                    path.specularBounce = true;
                }
                

                #if RUSSIAN_ROULETTE

                float maxBeta = glm::max(beta.x, glm::max(beta.y, beta.z));
                if (path.remainingBounces > 0 && maxBeta < 1 && depth > 1) {
                    #if HALTON_SEQUENCE
                    float rand = halton(iter, 29);
                    #else
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    float rand = u01(rng);
                    #endif
                    float result = glm::max(0.0f, 1.0f - maxBeta);
                    if (rand < result) {
                        path.remainingBounces = 0;
                    }
                    else {
                        beta /= 1 - result;
                    }
                }

                #endif
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            // some of the obj and gltf modelsim testing on dont have emmissive materials
            // so an environment light makes them visible
            #if ENVIRONMENT_LIGHT
                path.L += path.color * glm::vec3(0.8);
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
        image[iterationPath.pixelIndex] += iterationPath.L;
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
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_bvh_nodes,
            dev_bvh_geom_indices,
            num_bvh_nodes
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        #if CONTIGUOUS_BY_MATERIAL
        thrust::sort_by_key(
            thrust::device, dev_intersections, 
            dev_intersections + num_paths, dev_paths, 
            compareMaterial());
        #endif

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_tex_pixels,
            dev_light_geom_indices,
            num_lights,
            dev_bvh_nodes,
            dev_bvh_geom_indices,
            num_bvh_nodes
        );
        checkCUDAError("shade material");
        
        PathSegment* activeEnd = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, hasBouncesRemaining());
        num_paths = (int)(activeEnd - dev_paths);
        iterationComplete = !num_paths || depth >= traceDepth; // based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
