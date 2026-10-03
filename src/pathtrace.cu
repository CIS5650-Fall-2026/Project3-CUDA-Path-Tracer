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
#include <thrust/partition.h>
#include <thrust/sort.h>

#define SORTBYMAT 0
#define ERRORCHECK 0
#define USE_BVH 1

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

//taken from: https://www.shadertoy.com/view/3syfDc

__device__ glm::vec4 permute(glm::vec4 x)
{
    return glm::mod(((x * 34.0f) + 1.0f) * x, 289.0f);
}

__device__ glm::vec2 fade(glm::vec2 t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float cnoise(glm::vec2 P)
{
    glm::vec4 Pi = glm::floor(glm::vec4(P.x, P.y, P.x, P.y)) + glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    glm::vec4 Pf = glm::fract(glm::vec4(P.x, P.y, P.x, P.y)) - glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    Pi = glm::mod(Pi, 289.0f); // To avoid truncation effects in permutation
    glm::vec4 ix = glm::vec4(Pi.x, Pi.z, Pi.x, Pi.z);
    glm::vec4 iy = glm::vec4(Pi.y, Pi.y, Pi.w, Pi.w);
    glm::vec4 fx = glm::vec4(Pf.x, Pf.z, Pf.x, Pf.z);
    glm::vec4 fy = glm::vec4(Pf.y, Pf.y, Pf.w, Pf.w);
    glm::vec4 i = permute(permute(ix) + iy);
    glm::vec4 gx = 2.0f * glm::fract(i * 0.0243902439f) - 1.0f; // 1/41 = 0.024...
    glm::vec4 gy = glm::abs(gx) - 0.5f;
    glm::vec4 tx = glm::floor(gx + 0.5f);
    gx = gx - tx;
    glm::vec2 g00 = glm::vec2(gx.x, gy.x);
    glm::vec2 g10 = glm::vec2(gx.y, gy.y);
    glm::vec2 g01 = glm::vec2(gx.z, gy.z);
    glm::vec2 g11 = glm::vec2(gx.w, gy.w);
    glm::vec4 norm = 1.79284291400159f - 0.85373472095314f *
        glm::vec4(glm::dot(g00, g00), glm::dot(g01, g01), glm::dot(g10, g10), glm::dot(g11, g11));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;
    float n00 = glm::dot(g00, glm::vec2(fx.x, fy.x));
    float n10 = glm::dot(g10, glm::vec2(fx.y, fy.y));
    float n01 = glm::dot(g01, glm::vec2(fx.z, fy.z));
    float n11 = glm::dot(g11, glm::vec2(fx.w, fy.w));
    glm::vec2 fade_xy = fade(glm::vec2(Pf.x, Pf.y));
    glm::vec2 n_x = glm::mix(glm::vec2(n00, n01), glm::vec2(n10, n11), fade_xy.x);
    float n_xy = glm::mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3f * n_xy;
}

__device__ glm::vec3 sampleStarSky(glm::vec3 direction)
{
    glm::vec3 d = glm::normalize(direction);

    glm::vec2 uv(
        0.5f + atan2f(d.z, d.x) / (2.0f * PI),
        acosf(glm::clamp(d.y, -1.0f, 1.0f)) / PI
    );

    glm::vec2 cord = (uv - 0.5f) * 250.0f; 

    float b = cnoise(cord) + cnoise(cord * 2.0f) / 2.0f;

    b = (b - 1.0f) * 3.0f;

    b = glm::max(b, 0.0f);

    return glm::vec3(b, b, b);
}


__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__host__ __device__ float linearToSRGB(float value)
{
    value = glm::clamp(value, 0.0f, 1.0f);

    if (value <= 0.0031308f) {
        return 12.92f * value;
    }

    return 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / static_cast<float>(iter);

        glm::ivec3 color;
        color.x = static_cast<int>(linearToSRGB(pix.x) * 255.0f + 0.5f);
        color.y = static_cast<int>(linearToSRGB(pix.y) * 255.0f + 0.5f);
        color.z = static_cast<int>(linearToSRGB(pix.z) * 255.0f + 0.5f);

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
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static Triangle* dev_triangles = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static BVHNode* dev_bvhNodes = nullptr;
static PrimitiveRef* dev_bvhPrimitives = nullptr;
static Texture* dev_textures = nullptr;
static glm::vec3* dev_texturePixels = nullptr;
static DistantLight* dev_distantLights = nullptr;

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

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
   

    cudaMalloc((void**)&dev_bvhNodes, hst_scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, hst_scene->bvhNodes.data(), hst_scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_bvhPrimitives, hst_scene->bvhPrimitives.size() * sizeof(PrimitiveRef));
    cudaMemcpy(dev_bvhPrimitives, hst_scene->bvhPrimitives.data(), hst_scene->bvhPrimitives.size() * sizeof(PrimitiveRef), cudaMemcpyHostToDevice);
    

    cudaMalloc((void**)&dev_textures, hst_scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, hst_scene->textures.data(), hst_scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_texturePixels, hst_scene->texturePixels.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_texturePixels, hst_scene->texturePixels.data(), hst_scene->texturePixels.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_distantLights, scene->distantLights.size() * sizeof(DistantLight));

    cudaMemcpy(dev_distantLights, scene->distantLights.data(), scene->distantLights.size() * sizeof(DistantLight), cudaMemcpyHostToDevice);
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
    cudaFree(dev_bvhPrimitives);
    cudaFree(dev_textures);
    cudaFree(dev_texturePixels);
    cudaFree(dev_distantLights);

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
        segment.radiance = glm::vec3(0.0f);

        //MAIN IDEA for antialiasing: we combine our samples across 4 iterations for antialiasing

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        //dividing our pixels into 4 subpixels, each iteration we will use the same subpixel
        int cell = (iter - 1) % 4;

        //acquires our subpixels loc
        int cellX = cell % 2;
        int cellY = cell / 2;

        //makes each subpixel half a pixel wide and centers the complete sampling region around the existing pixel locs
        float offsetX = (cellX + u01(rng)) * 0.5f - 0.5f;
        float offsetY = (cellY + u01(rng)) * 0.5f - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x  + offsetX- (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + offsetY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

//seeing if we get an object that is closer but still in front of our ray 
//if it is valid, we set the closestT and the UV and normal
__device__ void testTriangleAndUpdateHit(
    const Triangle& triangle,
    const Ray& ray,
    float& closestT,
    int& hitMaterialId,
    glm::vec3& hitNormal,
    glm::vec2& hitUV,
    glm::vec3& hitDpdu,
    glm::vec3& hitDpdv,
    glm::vec3& hitDndu,
    glm::vec3& hitDndv)
{
    glm::vec2 barycentric;
    float t = triangleIntersectionTest(triangle, ray, barycentric);

    //behind or farther than closest object
    if (t <= 0.0f || t >= closestT) 
        return;
    
    closestT = t;
    hitMaterialId = triangle.materialId;
    hitDpdu = glm::vec3(0.0f);
    hitDpdv = glm::vec3(0.0f);
    hitDndu = glm::vec3(0.0f);
    hitDndv = glm::vec3(0.0f);

    float weightB = barycentric.x;
    float weightC = barycentric.y;
    float weightA = 1.0f - weightB - weightC;

    hitUV = weightA * triangle.uvs[0] + weightB * triangle.uvs[1] + weightC * triangle.uvs[2];

    hitNormal = glm::normalize(weightA * triangle.normals[0] + weightB * triangle.normals[1] + weightC * triangle.normals[2]);

    if (glm::dot(hitNormal, ray.direction) > 0.0f) 
        hitNormal = -hitNormal;

}

__device__ void testGeomAndUpdateHit(
    const Geom& geom,
    const Ray& ray,
    float& t_min,
    int& hitMaterialId,
    glm::vec3& normal,
    glm::vec2& hitUV,
    glm::vec3& hitDpdu,
    glm::vec3& hitDpdv,
    glm::vec3& hitDndu,
    glm::vec3& hitDndv)
{
    float t = -1.0f;
    bool outside = true;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    if (geom.type == CUBE)
    {
        t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
    }
    else if (geom.type == SPHERE)
    {
        t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
    }

    //compute the minimum t from the intersection tests to determine what
    //scene geometry object was hit first
    if (t > 0.0f && t_min > t)
    {
        t_min = t;
        hitMaterialId = geom.materialid;
        normal = tmp_normal;

        hitDpdu = glm::vec3(0.0f);
        hitDpdv = glm::vec3(0.0f);
        hitDndu = glm::vec3(0.0f);
        hitDndv = glm::vec3(0.0f);

        hitUV = glm::vec2(0.0f);

        if (geom.type == SPHERE) {
            //local space
            glm::vec3 localPoint = glm::vec3(geom.inverseTransform * glm::vec4(tmp_intersect, 1.0f));

            //original normal
            glm::vec3 localNormal = glm::normalize(localPoint);

            const float pi = 3.14159265358979323846f;

            float u = 0.5f + atan2f(localNormal.z, localNormal.x) / (2.0f * pi);
            float v = acosf(glm::clamp(localNormal.y, -1.0f, 1.0f)) / pi;

            hitUV = glm::vec2(u, v);

            float phi = 2.0f * pi * (u - 0.5f);
            float theta = pi * v;

            //the original radius of the sphere is 0.5, so we use that in our local space
            const float radius = 0.5f;

            glm::vec3 localDpdu = 2.0f * pi * radius * glm::vec3(-sinf(theta) * sinf(phi), 0.0f, sinf(theta) * cosf(phi));

            glm::vec3 localDpdv = pi * radius * glm::vec3(cosf(theta) * cosf(phi), -sinf(theta), cosf(theta) * sinf(phi));

            glm::vec3 localDndu = localDpdu / radius;
            glm::vec3 localDndv = localDpdv / radius;

            //transform directions into world space
            hitDpdu = glm::vec3(geom.transform * glm::vec4(localDpdu, 0.0f));

            hitDpdv = glm::vec3(geom.transform * glm::vec4(localDpdv, 0.0f));

            //get the dndu and dndv into world space

            //transform the original local space unit normal
            glm::vec3 transformedNormal = glm::vec3(geom.invTranspose * glm::vec4(localNormal, 0.0f));

            float normalLength = glm::length(transformedNormal);
            glm::vec3 worldNormal = transformedNormal / normalLength;

            //transform the local normal derivatives
            glm::vec3 transformedDndu = glm::vec3(geom.invTranspose * glm::vec4(localDndu, 0.0f));

            glm::vec3 transformedDndv = glm::vec3(geom.invTranspose * glm::vec4(localDndv, 0.0f));

            //accounting for normalizing the transformed normal
            hitDndu = (transformedDndu - worldNormal * glm::dot(worldNormal, transformedDndu)) / normalLength;

            hitDndv = (transformedDndv - worldNormal * glm::dot(worldNormal, transformedDndv)) / normalLength;

            //match the normal flip performed for hits from inside the sphere
            if (!outside) {
                hitDndu = -hitDndu;
                hitDndv = -hitDndv;
            }
        }
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
    const Triangle* triangles,
    int triangleCount,
    const BVHNode* bvhNodes,
    const PrimitiveRef* bvhPrimitives,
    int bvhNodeCount,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hitMaterialId = -1;

        glm::vec2 hitUV(0.0f);

        glm::vec3 hitDpdu(0.0f);
        glm::vec3 hitDpdv(0.0f);
        glm::vec3 hitDndu(0.0f);
        glm::vec3 hitDndv(0.0f);

        //USING BVH
        if (USE_BVH && bvhNodeCount > 0) {
            int stack[64];
            int stackSize = 0;

            stack[stackSize++] = 0;

            while (stackSize > 0) {
                int nodeIndex = stack[--stackSize];
                const BVHNode& node = bvhNodes[nodeIndex];

                if (!boundingBoxIntersectionTest(node.minCorner, node.maxCorner, pathSegment.ray, t_min)) {
                    continue;
                }

                if (node.primitiveCount > 0) {
                    for (int offset = 0; offset < node.primitiveCount; offset++) {
                        const PrimitiveRef& primitive = bvhPrimitives[node.firstPrimitive + offset];

                        if (primitive.type == PRIMITIVE_TRIANGLE) {
                            testTriangleAndUpdateHit(
                                triangles[primitive.index],
                                pathSegment.ray,
                                t_min,
                                hitMaterialId,
                                normal,
                                hitUV,
                                hitDpdu,
                                hitDpdv,
                                hitDndu,
                                hitDndv
                            );
                        }
                        else if (primitive.type == PRIMITIVE_GEOM) {
                            testGeomAndUpdateHit(
                                geoms[primitive.index],
                                pathSegment.ray,
                                t_min,
                                hitMaterialId,
                                normal,
                                hitUV,
                                hitDpdu,
                                hitDpdv,
                                hitDndu,
                                hitDndv
                            );
                        }
                    }
                }
                else {
                    //since we got to an internal node, we save both children for later testing.
                    stack[stackSize++] = node.rightChild;
                    stack[stackSize++] = node.leftChild;
                }
            }
        }
        //Not using BVH
        else {

            for (int i = 0; i < geoms_size; i++)
            {
                testGeomAndUpdateHit(
                    geoms[i],
                    pathSegment.ray,
                    t_min,
                    hitMaterialId,
                    normal,
                    hitUV,
                    hitDpdu,
                    hitDpdv,
                    hitDndu,
                    hitDndv
                );
            }

            for (int i = 0; i < triangleCount; i++) {
                testTriangleAndUpdateHit(
                    triangles[i],
                    pathSegment.ray,
                    t_min,
                    hitMaterialId,
                    normal,
                    hitUV,
                    hitDpdu,
                    hitDpdv,
                    hitDndu,
                    hitDndv
                );
            }
        }

        if (hitMaterialId == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = -1;
            intersections[path_index].uv = glm::vec2(0.0f);

            intersections[path_index].dpdu = glm::vec3(0.0f);
            intersections[path_index].dpdv = glm::vec3(0.0f);

        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hitMaterialId;
            intersections[path_index].surfaceNormal = normal;

            intersections[path_index].uv = hitUV;

            intersections[path_index].dpdu = hitDpdu;
            intersections[path_index].dpdv = hitDpdv;
            intersections[path_index].dndu = hitDndu;
            intersections[path_index].dndv = hitDndv;
        }
    }
}

__device__ glm::vec3 sampleTexture(
    const Texture& texture,
    const glm::vec3* pixels,
    glm::vec2 uv)
{
    //wrapping
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);

    int x = static_cast<int>(u * texture.width);
    int y = static_cast<int>(v * texture.height);

    //guards against rounding at the image boundary.
    if (x >= texture.width) 
        x = texture.width - 1;
    if (y >= texture.height) 
        y = texture.height - 1;
    
    int pixelIndex = texture.pixelOffset + y * texture.width + x;

    return pixels[pixelIndex];
}

//just a checkerboard
__device__ glm::vec3 proceduralTexture(const glm::vec2& uv)
{
    int columns = 16;
    int rows = 8;

    float u = uv.x - floorf(uv.x);
    float v = glm::clamp(uv.y, 0.0f, 1.0f);

    int column = static_cast<int>(floorf(u * columns));
    int row = static_cast<int>(floorf(v * rows));

    if (row >= rows) {
        row = rows - 1;
    }

    if (((column + row) % 2) != 0) {
        //green
        return glm::vec3(0.1f, 0.8f, 0.2f);
    }
    else {
        //pink
        return glm::vec3(1.0f, 0.2f, 0.5f);
    }

}

//note this is only for spheres!
__device__ float sampleHeight(
    const Texture& texture,
    const glm::vec3* pixels,
    glm::vec2 uv)
{
    //we wrap horizontally around the sphere and
    // clamp at its poles
    float u = uv.x - floorf(uv.x);
    float v = glm::clamp(uv.y, 0.0f, 1.0f);

    //convert uv into pixel coords
    float x = u * texture.width - 0.5f;
    float y = v * texture.height - 0.5f;
    
    //since our uv will most likely not hit an exact pixel coordinate, we use bilinear interpolation with the 4 neighboring pixels 
    int x0 = static_cast<int>(glm::floor(x));
    int y0 = static_cast<int>(glm::floor(y));

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    //we wrap x0 and x1 since it goes around the sphere
    x0 = (x0 % texture.width + texture.width) % texture.width;
    x1 = (x1 % texture.width + texture.width) % texture.width;

    //we clamp y0 and y1 since it goes from the north to the south pole
    y0 = glm::clamp(y0, 0, texture.height - 1);
    y1 = glm::clamp(y1, 0, texture.height - 1);

    //get the height of each of these pixels
    float h00 = pixels[texture.pixelOffset + y0 * texture.width + x0].r;
    float h10 = pixels[texture.pixelOffset + y0 * texture.width + x1].r;
    float h01 = pixels[texture.pixelOffset + y1 * texture.width + x0].r;
    float h11 = pixels[texture.pixelOffset + y1 * texture.width + x1].r;

    //get out t vals
    float tx = x - glm::floor(x);
    float ty = y - glm::floor(y);

    //interpolate
    float top = h00 * (1.0f - tx) + h10 * tx;
    float bottom = h01 * (1.0f - tx) + h11 * tx;

    return top * (1.0f - ty) + bottom * ty;
}


__device__ float intersectPrimitive(
    const PrimitiveRef& primitive,
    const Ray& ray,
    const Geom* geoms,
    const Triangle* triangles)
{
    if (primitive.type == PRIMITIVE_TRIANGLE) {
        glm::vec2 barycentric;
        return triangleIntersectionTest(triangles[primitive.index], ray, barycentric);
    }

    const Geom& geom = geoms[primitive.index];

    glm::vec3 hitPoint;
    glm::vec3 normal;
    bool outside;

    if (geom.type == CUBE) {
        return boxIntersectionTest(geom, ray, hitPoint, normal, outside);
    }

    if (geom.type == SPHERE) {
        return sphereIntersectionTest(geom, ray, hitPoint, normal, outside);
    }

    return -1.0f;
}

__device__ bool isOccluded(
    const Ray& ray,
    float maxDistance,
    const Geom* geoms,
    int geomCount,
    const Triangle* triangles,
    int triangleCount,
    const BVHNode* bvhNodes,
    const PrimitiveRef* bvhPrimitives,
    int bvhNodeCount)
{
    if (USE_BVH && bvhNodeCount > 0) {
        int stack[64];
        int stackSize = 0;
        stack[stackSize++] = 0;

        while (stackSize > 0) {
            int nodeIndex = stack[--stackSize];
            const BVHNode& node = bvhNodes[nodeIndex];

            //skip the node if the shadow ray misses its bounds
            if (!boundingBoxIntersectionTest(
                node.minCorner,
                node.maxCorner,
                ray,
                maxDistance)) {

                continue;
            }

            if (node.primitiveCount > 0) {
                //we are at a leaf, so we test the primitive
                for (int offset = 0; offset < node.primitiveCount; ++offset) {
                    const PrimitiveRef& primitive = bvhPrimitives[node.firstPrimitive + offset];

                    float t = intersectPrimitive(primitive, ray, geoms, triangles);

                    if (t > 0.0f && t < maxDistance) {
                        return true;
                    }
                }
            }
            else {
                //visit the children nodes
                stack[stackSize++] = node.rightChild;
                stack[stackSize++] = node.leftChild;
            }
        }
    }
    else {
        for (int i = 0; i < geomCount; ++i) {
            PrimitiveRef primitive{ PRIMITIVE_GEOM, i };

            float t = intersectPrimitive(primitive, ray, geoms, triangles);

            if (t > 0.0f && t < maxDistance) {
                return true;
            }
        }

        for (int i = 0; i < triangleCount; ++i) {
            PrimitiveRef primitive{ PRIMITIVE_TRIANGLE, i };

            float t = intersectPrimitive(primitive, ray, geoms, triangles);

            if (t > 0.0f && t < maxDistance) {
                return true;
            }
        }
    }

    return false;
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
    const Texture* textures,
    const glm::vec3* texturePixels,
    const DistantLight* distantLights,
    int distantLightCount,
    const Geom* geoms,
    int geomCount,
    const Triangle* triangles,
    int triangleCount,
    const BVHNode* bvhNodes,
    const PrimitiveRef* bvhPrimitives,
    int bvhNodeCount,
    float sceneRadius)
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
            
            //got a hit, get the color if the object hit has a texture

            //use our procedurally made texture if the object hit uses it
            if (material.useCheckerboard) {

                material.color = proceduralTexture(intersection.uv);
            }
            //non zero means we have used a texture
            else if (material.colorTextureId >= 0) {
                const Texture& texture = textures[material.colorTextureId];

                material.color *= sampleTexture( texture, texturePixels, intersection.uv);
            }
            
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].radiance += pathSegments[idx].color * materialColor * material.emittance;
                //it hit a light, so should terminate
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {

                glm::vec3 hitPoint = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

                glm::vec3 shadingNormal = intersection.surfaceNormal;

                if (material.bumpTextureId >= 0) {
                    const Texture& heightTexture = textures[material.bumpTextureId];

                    //referenced pbrt 

                    //basic assumption that du and dv are just 0.01 since I didn't implement 
                    float du = 0.01;
                    float dv = 0.01;

                    float displace = material.bumpStrength * sampleHeight(heightTexture, texturePixels, intersection.uv);

                    float uDisplace = material.bumpStrength * sampleHeight(heightTexture, texturePixels, intersection.uv + glm::vec2(du, 0.0f));
                    float vDisplace = material.bumpStrength * sampleHeight(heightTexture, texturePixels, intersection.uv + glm::vec2(0.0f, dv));

                    glm::vec3 newDpdu = intersection.dpdu +
                    (uDisplace - displace) / du * intersection.surfaceNormal +
                    displace * intersection.dndu;
                    glm::vec3 newDpdv = intersection.dpdv +
                    (vDisplace - displace) / dv * intersection.surfaceNormal +
                    displace * intersection.dndv;

                    glm::vec3 bumpedNormal = glm::cross(newDpdu, newDpdv);

                    //make sure we dont have zero length or parallel tangents
                    if (glm::length(newDpdu) * glm::length(newDpdv) > 0.0 &&
                        glm::length(bumpedNormal) > EPSILON * glm::length(newDpdu) * glm::length(newDpdv)) {

                        bumpedNormal = glm::normalize(bumpedNormal);

                        //make sure our bumped normal is in the same hemisphere as our og normal
                        if (glm::dot(bumpedNormal, intersection.surfaceNormal) < 0.0f) {
                            bumpedNormal = -bumpedNormal;
                        }

                        shadingNormal = bumpedNormal;
                    }
                       
                }

                PathSegment& path = pathSegments[idx];

                glm::vec3 offsetNormal = glm::normalize(intersection.surfaceNormal);
                shadingNormal = glm::normalize(shadingNormal);

                //referenced: https://docs.unity.cn/430/Documentation/Components/SL-SurfaceShaderExamples.html 
                //for rim lighting
                if (material.atmosphereStrength > 0.0f) {
                    glm::vec3 viewDir = glm::normalize(-path.ray.direction);
                    glm::vec3 rimColor(0.08f, 0.65f, 1.0f);
                    float rimPower = 8.0f;

                    float rim = 1.0f - glm::clamp(glm::dot(viewDir, offsetNormal), 0.0f, 1.0f);

                    glm::vec3 emission = rimColor * powf(rim, rimPower);
                    emission *= material.atmosphereStrength;

                    path.radiance += path.color * emission;
                }

                //shoot out our shadow ray to our sunlight

                //this is the scene diameter with a small allowance for the origin offset
                float shadowDistance = 2.0f * sceneRadius + EPSILON;

                for (int lightIndex = 0; lightIndex < distantLightCount; ++lightIndex) {
                    if (material.hasReflective > 0.0f) {
                        break;
                    }
                    const DistantLight& light = distantLights[lightIndex];
                    
                    //specific edit so that my directional lights (other than my general sun lighting) affects the scene
                    if (!material.receiveAccentLights && lightIndex > 0) {
                        continue;
                    }

                    glm::vec3 wi = light.toLight;
                    float cosTheta = glm::dot(shadingNormal, wi);

                    //we skip light below either the shading or original surface normal
                    if (cosTheta <= 0.0f || glm::dot(offsetNormal, wi) <= 0.0f) {
                        continue;
                    }

                    Ray shadowRay;
                    shadowRay.origin = hitPoint + offsetNormal * EPSILON;
                    shadowRay.direction = wi;

                    bool blocked = isOccluded(
                        shadowRay,
                        shadowDistance,
                        geoms,
                        geomCount,
                        triangles,
                        triangleCount,
                        bvhNodes,
                        bvhPrimitives,
                        bvhNodeCount
                    );

                    if (!blocked) {
                        glm::vec3 brdf = material.color / PI;

                        path.radiance += path.color * brdf * light.strength * cosTheta * material.lightingMultiplier;
                    }
                }

                //continue indirect lighting only if another bounce is allowed
                path.remainingBounces--;

                if (path.remainingBounces > 0) {
                    scatterRay(path, hitPoint, shadingNormal, material, rng);
                }
                
              
                /*
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not*/
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            //hit nothing, should terminate
            PathSegment& path = pathSegments[idx];

            path.radiance += path.color * sampleStarSky(path.ray.direction);
            path.remainingBounces = 0;
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
        image[iterationPath.pixelIndex] += iterationPath.radiance;
    }
}

struct IsActivePath {
    __host__ __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

struct CompareMaterial {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
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

    while (num_paths > 0)
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
            dev_triangles,
            hst_scene->triangles.size(),
            dev_bvhNodes,
            dev_bvhPrimitives,
            static_cast<int>(hst_scene->bvhNodes.size()),
            dev_intersections
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

        //sort by material now
        if (SORTBYMAT) {
            thrust::sort_by_key(
                thrust::device,
                dev_intersections,
                dev_intersections + num_paths,
                dev_paths,
                CompareMaterial{}
            );
        }

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_texturePixels,
            dev_distantLights,
            static_cast<int>(hst_scene->distantLights.size()),
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            dev_triangles,
            static_cast<int>(hst_scene->triangles.size()),
            dev_bvhNodes,
            dev_bvhPrimitives,
            static_cast<int>(hst_scene->bvhNodes.size()),
            hst_scene->sceneRadius
        );

        //sort so that all active paths are in the front, and all finished paths at the end
        dev_path_end = thrust::partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            IsActivePath{}
        );

        //getting number of active paths
        num_paths = static_cast<int>(dev_path_end - dev_paths);

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
