#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

/**
 * Test intersection between a ray and a triangle in the scene. 
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float triangleIntersectionTest(Geom tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside);

    // tests intersection of the ray and the bounding box.
__host__ __device__ float bboxIntersectionTest(BVHBounds bbox, Ray r);
    /**
 * Test intersection between a ray and the bvh node, ultimately returns teh intersection of the target primitive (if there is any).
 *
* @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 *  @param geomIdx            Output param for the index of the geometry we intersect

 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float bvhNodeIntersectionTest(
    int bvhnodeIdx, 
    BVHNode* bvhnodes,
    int bvhnodes_size,

    Geom* geoms,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside,
    int& geomIdx);
/**
 * Handles all geometry intersection, calls intersection tests for sphere and box and triangles or whatever
 *
* @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @param hit_geom_index    Output param for the exact geometry index that we hit
 * @return                   Ray parameter `t` value. FLT_MAX if no intersection, in which case hit_geom_index is unmodified (-1).
*/
 __host__ __device__ float geometryIntersectionTest(
    Geom* geoms,
    int geoms_size,
    BVHNode* bvhnodes,
    int bvhnodes_size,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    int& hit_geom_index
);

/**
 * checks if we intersect a plane which is the area light
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside (or behind the light).
 * @return                   Ray parameter `t` value. -1 if no intersection.
 * */
__host__ __device__ float areaLightIntersectionTest(
    Light light,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside);

/**
 * Handles all light intersection, which is just area light at this point in time
 *
* @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param hit_light_index    Output param for the exact light index that we hit
 * @return                   Ray parameter `t` value.  FLT_MAX if no intersection, in which case hit_light_index is unmodified (-1).
*/
__host__ __device__ float lightIntersectionTest(
    Light* lights,
    int lights_size,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    int& hit_light_index
);