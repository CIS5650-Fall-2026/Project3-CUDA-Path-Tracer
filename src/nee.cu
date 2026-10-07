#include "nee.h"

#include "intersections.h"

namespace Nee
{

__device__ float power_heuristic(float pdfA, float pdfB) {
    float pdfA2 = pdfA * pdfA;
    float pdfB2 = pdfB * pdfB;
    float denom = pdfA2 + pdfB2;
    return denom > 0.0f ? pdfA2 / denom : 0.0f;
}

__device__ float shadow_intersect_aabb(const Ray& ray, const BVHNode& node,
                                       float maxT) {
    float tmin = 0.0f;
    float tmax = maxT;

    for (int axis = 0; axis < 3; ++axis) {
        float origin = ray.origin[axis];
        float direction = ray.direction[axis];
        float minBound = node.aabbMin[axis];
        float maxBound = node.aabbMax[axis];

        if (fabsf(direction) < 0.0000001f) {
            if (origin < minBound || origin > maxBound) {
                return FLT_MAX;
            }
            continue;
        }

        float invDir = 1.0f / direction;
        float t1 = (minBound - origin) * invDir;
        float t2 = (maxBound - origin) * invDir;
        tmin = fmaxf(tmin, fminf(t1, t2));
        tmax = fminf(tmax, fmaxf(t1, t2));
    }

    return tmax >= tmin && tmin < maxT ? tmin : FLT_MAX;
}

__device__ bool shadow_bvh_occluded(const Ray& shadowRay,
                                    Triangle* triangles,
                                    int triangles_size,
                                    BVHNode* bvh,
                                    Material* materials,
                                    int materials_size,
                                    float maxDistance) {
    if (triangles == nullptr || triangles_size <= 0 || bvh == nullptr) {
        return false;
    }

    int nodeIdx = 0;
    constexpr int STACK_SIZE = 128;
    int stack[STACK_SIZE];
    int stackPtr = 0;

    while (true) {
        BVHNode& node = bvh[nodeIdx];

        if (node.isLeaf()) {
            for (uint32_t i = 0; i < node.triCount; ++i) {
                int triangleIdx = (int)(node.firstTriIdx + i);
                if (triangleIdx >= triangles_size) {
                    continue;
                }
                int materialId = triangles[triangleIdx].materialid;
                if (materialId >= 0 && materialId < materials_size &&
                    materials[materialId].transmission_factor > 0.0f) {
                    continue;
                }

                glm::vec3 intersectPoint;
                glm::vec3 normal;
                bool outside = true;
                float t = triangleIntersectionTest(
                    triangles[triangleIdx], shadowRay, intersectPoint, normal,
                    outside);
                if (t > 0.0002f && t < maxDistance) {
                    return true;
                }
            }

            if (stackPtr == 0) {
                break;
            }
            nodeIdx = stack[--stackPtr];
            continue;
        }

        int child1 = (int)node.leftNode;
        int child2 = child1 + 1;
        float dist1 = shadow_intersect_aabb(shadowRay, bvh[child1],
                                            maxDistance);
        float dist2 = shadow_intersect_aabb(shadowRay, bvh[child2],
                                            maxDistance);

        if (dist1 > dist2) {
            float tempDist = dist1;
            dist1 = dist2;
            dist2 = tempDist;

            int tempChild = child1;
            child1 = child2;
            child2 = tempChild;
        }

        if (dist1 == FLT_MAX) {
            if (stackPtr == 0) {
                break;
            }
            nodeIdx = stack[--stackPtr];
        } else {
            nodeIdx = child1;
            if (dist2 != FLT_MAX && stackPtr < STACK_SIZE) {
                stack[stackPtr++] = child2;
            }
        }
    }

    return false;
}

__device__ bool shadow_ray_visible(const glm::vec3& origin,
                                   const NeeSample& neeSample,
                                   Geom* geoms,
                                   int geoms_size,
                                   Triangle* triangles,
                                   int triangles_size,
                                   BVHNode* bvh,
                                   Material* materials,
                                   int materials_size) {
    float maxDistance = neeSample.distance - 2.0f * 0.0002f;
    if (maxDistance <= 0.0002f) {
        return false;
    }

    Ray shadowRay;
    shadowRay.origin = origin + 0.0002f * neeSample.wi;
    shadowRay.direction = neeSample.wi;

    for (int i = 0; i < geoms_size; ++i) {
        Geom& geom = geoms[i];
        if (geom.materialid >= 0 && geom.materialid < materials_size &&
            materials[geom.materialid].transmission_factor > 0.0f) {
            continue;
        }
        glm::vec3 intersectPoint;
        glm::vec3 normal;
        bool outside = true;
        float t = -1.0f;
        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, shadowRay, intersectPoint, normal,
                                    outside);
        } else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, shadowRay, intersectPoint, normal,
                                       outside);
        }

        if (t > 0.0002f && t < maxDistance) {
            return false;
        }
    }

    return !shadow_bvh_occluded(shadowRay, triangles, triangles_size, bvh,
                                materials, materials_size, maxDistance);
}

__device__ glm::vec3 evaluate_emissive_hit(
    const PathSegment& pathSegment,
    glm::vec3 hitPoint,
    glm::vec3 lightNormal,
    int triangleId,
    const Material& lightMaterial,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    float totalLightArea) {
    float misWeight = 1.0f;
    if (!pathSegment.lastBounceWasSpecular &&
        pathSegment.lastBsdfPdf > 0.0f &&
        triangleId >= 0) {
        float lightPdf = pdf_light_for_triangle_hit(
            pathSegment.ray.origin, hitPoint, lightNormal, triangleId, lights,
            lights_size, triangles, triangles_size, lightMaterial,
            totalLightArea);
        misWeight = power_heuristic(pathSegment.lastBsdfPdf, lightPdf);
    }

    return misWeight * lightMaterial.emissive_factor;
}

__device__ LightSample sample_light(
    SceneLight* lights,
    int lights_size,
    float totalLightArea,
    thrust::default_random_engine& rng)
{
    LightSample sample{};
    sample.lightIndex = -1;
    sample.lightPickPdf = 0.0f;
    sample.valid = false;

    if (lights == nullptr || lights_size <= 0 || totalLightArea <= 0.0f)
    {
        return sample;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float targetArea = u01(rng) * totalLightArea;

    float accumulatedArea = 0.0f;
    for (int i = 0; i < lights_size; i++)
    {
        if (lights[i].type != SCENE_LIGHT_TRIANGLE || lights[i].area <= 0.0f)
        {
            continue;
        }

        accumulatedArea += lights[i].area;

        if (accumulatedArea >= targetArea)
        {
            sample.light = lights[i];
            sample.lightIndex = i;
            sample.lightPickPdf = lights[i].area / totalLightArea;
            sample.valid = true;
            return sample;
        }
    }

    return sample;
}

__device__ LightPointSample sample_point_from_light(
    const LightSample& lightSample,
    Triangle* triangles,
    int triangles_size,
    thrust::default_random_engine& rng)
{
    LightPointSample sample{};
    sample.materialId = -1;
    sample.pdfArea = 0.0f;
    sample.valid = false;

    if (!lightSample.valid ||
        lightSample.light.type != SCENE_LIGHT_TRIANGLE ||
        triangles == nullptr ||
        lightSample.light.id < 0 ||
        lightSample.light.id >= triangles_size)
    {
        return sample;
    }

    const Triangle& triangle = triangles[lightSample.light.id];
    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;
    glm::vec3 geometricNormal = glm::cross(edge1, edge2);
    float normalLength = glm::length(geometricNormal);
    if (normalLength <= 0.0f)
    {
        return sample;
    }

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float r1 = sqrtf(u01(rng));
    float r2 = u01(rng);
    float b0 = 1.0f - r1;
    float b1 = r1 * (1.0f - r2);
    float b2 = r1 * r2;

    sample.position = b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;
    sample.normal = b0 * triangle.n0 + b1 * triangle.n1 + b2 * triangle.n2;
    if (glm::dot(sample.normal, sample.normal) <= 0.0f)
    {
        sample.normal = geometricNormal;
    }
    sample.normal = glm::normalize(sample.normal);
    sample.barycentric = glm::vec3(b0, b1, b2);
    sample.materialId = triangle.materialid;
    sample.pdfArea = 1.0f / lightSample.light.area;
    sample.valid = true;
    return sample;
}

__device__ NeeSample get_nee(
    const glm::vec3& surfacePoint,
    const glm::vec3& surfaceNormal,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    Material* materials,
    int materials_size,
    float totalLightArea,
    thrust::default_random_engine& rng)
{
    NeeSample result{};
    result.distance = 0.0f;
    result.pdfArea = 0.0f;
    result.pdfDirectional = 0.0f;
    result.pdfLight = 0.0f;
    result.lightIndex = -1;
    result.materialId = -1;
    result.valid = false;

    LightSample lightSample =
        sample_light(lights, lights_size, totalLightArea, rng);
    LightPointSample pointSample =
        sample_point_from_light(lightSample, triangles, triangles_size, rng);
    if (!lightSample.valid || !pointSample.valid ||
        pointSample.materialId < 0 || pointSample.materialId >= materials_size ||
        materials == nullptr)
    {
        return result;
    }

    glm::vec3 toLight = pointSample.position - surfacePoint;
    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= 0.0f)
    {
        return result;
    }

    float distance = sqrtf(distanceSquared);
    glm::vec3 wi = toLight / distance;
    float cosLight = glm::dot(pointSample.normal, -wi);
    const Material& lightMaterial = materials[pointSample.materialId];
    if (lightMaterial.double_sided)
    {
        cosLight = fabsf(cosLight);
    }
    else
    {
        cosLight = fmaxf(cosLight, 0.0f);
    }
    if (cosLight <= 0.0f || lightSample.lightPickPdf <= 0.0f ||
        pointSample.pdfArea <= 0.0f)
    {
        return result;
    }

    float cosSurface = fmaxf(glm::dot(surfaceNormal, wi), 0.0f);
    if (cosSurface <= 0.0f)
    {
        return result;
    }

    result.position = pointSample.position;
    result.normal = pointSample.normal;
    result.wi = wi;
    result.Li = lightMaterial.emissive_factor;
    result.distance = distance;
    result.pdfArea = pointSample.pdfArea;
    result.pdfDirectional = pointSample.pdfArea * distanceSquared / cosLight;
    result.pdfLight = lightSample.lightPickPdf * result.pdfDirectional;
    result.lightIndex = lightSample.lightIndex;
    result.materialId = pointSample.materialId;
    result.valid = result.pdfLight > 0.0f &&
                   glm::dot(result.Li, result.Li) > 0.0f;
    return result;
}

__device__ float pdf_light_for_triangle_hit(
    const glm::vec3& surfacePoint,
    const glm::vec3& lightPoint,
    const glm::vec3& lightNormal,
    int triangleId,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    const Material& lightMaterial,
    float totalLightArea)
{
    if (triangleId < 0 || triangleId >= triangles_size ||
        triangles == nullptr || lights == nullptr || lights_size <= 0 ||
        totalLightArea <= 0.0f)
    {
        return 0.0f;
    }

    SceneLight matchedLight{};
    bool foundLight = false;
    for (int i = 0; i < lights_size; ++i)
    {
        if (lights[i].type != SCENE_LIGHT_TRIANGLE ||
            lights[i].id != triangleId)
        {
            continue;
        }
        matchedLight = lights[i];
        foundLight = true;
        break;
    }
    if (!foundLight || matchedLight.area <= 0.0f)
    {
        return 0.0f;
    }

    glm::vec3 toLight = lightPoint - surfacePoint;
    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= 0.0f)
    {
        return 0.0f;
    }

    glm::vec3 wi = glm::normalize(toLight);
    float cosLight = glm::dot(lightNormal, -wi);
    cosLight = lightMaterial.double_sided ? fabsf(cosLight)
                                          : fmaxf(cosLight, 0.0f);
    if (cosLight <= 0.0f)
    {
        return 0.0f;
    }

    float pdfArea = 1.0f / matchedLight.area;
    float pdfDirectional = pdfArea * distanceSquared / cosLight;
    float lightPickPdf = matchedLight.area / totalLightArea;
    return lightPickPdf * pdfDirectional;
}

}
