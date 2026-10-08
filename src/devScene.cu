#include "devScene.h"

#include "material.h"
#include "samplers.h"
#include "intersection.h"

__device__ IntersectionData DevScene::intersect(const Ray& ray) {
    IntersectionData result = IntersectionData{};

    int tlasIterationCount = 0;
    int blasIterationCount = 0;

    int nodeStack[32]; // We limit the depth of the tree when building it to not surpass this stack
    int stackPtr = 0;

    nodeStack[stackPtr++] = 0; // tlas node index 0 is always the root

    while (stackPtr > 0) {
        tlasIterationCount++;

        // Pop node off stack
        int nodeIndex = nodeStack[--stackPtr];
        const TLASNode& node = dev_tlasNodes[nodeIndex];

        float tNear;
        if (!node.intersect(ray, tNear)) {
            // Missed this node, skip it
            continue;
        }

        if (result.t > 0.0f && tNear > result.t) {
            // We already have a closer intersection, skip this node
            continue;
        }

        if (node.leftChild == -1) {
            // Hit leaf node, check geometry
			int geomIndex = node.geometryIndex;
            if (geomIndex < 0 || geomIndex >= m_geometryCount) {
                continue;
            }

			Geom geometry = dev_geometry[geomIndex];
			IntersectionData geometryResult = IntersectionStatics::intersectGeometry(ray, geometry, dev_blasNodes, dev_triangles, m_lightCount);
            blasIterationCount += geometryResult.blasIterationCount;

            if(geometryResult.t > 0.0f && (result.t < 0.0f || geometryResult.t < result.t)) {
                result = geometryResult;
                result.geometryIndex = geomIndex;
			}
        }
        else {
            // Push children onto stack if there is room
            if (stackPtr + 2 < 32) {
                nodeStack[stackPtr++] = node.leftChild;
                nodeStack[stackPtr++] = node.leftChild + 1;
            }
        }
    }

    result.tlasIterationCount = tlasIterationCount;
    result.blasIterationCount = blasIterationCount;

    return result;
}

__device__ glm::vec3 DevScene::nextEventEstimation(const glm::vec4& random, 
    const Ray& incomingRay, 
    const IntersectionData& intersectionData,
	glm::vec3& outDirectionToLight,
    float& outPdf)
{
	outDirectionToLight = glm::vec3(0.0f);
    outPdf = 0.0f;

    if (m_lightCount == 0) {
        return glm::vec3(0.0f);
    }

    // Pick a random light
    int chosenLightIndex = (int)(random.w * (float)m_lightCount);
    if (chosenLightIndex < 0 || chosenLightIndex >= m_lightCount) {
        return glm::vec3(0.0f);
    }
    float lightIndexPdf = 1.0 / (float)m_lightCount;

    Geom lightGeometry = dev_geometry[chosenLightIndex];
    if (lightGeometry.materialid >= m_materialCount) {
        return glm::vec3(0.0f);
    }

    Material lightMaterial = dev_materials[lightGeometry.materialid];
    if (lightMaterial.emittance <= 0.0f) {
        return glm::vec3(0.0f);
    }

    // Sample a point on the light
    glm::vec3 lightNormal = glm::vec3(0.0f);
    float lightSurfacePdf = 1.0f;
    glm::vec3 lightPosition = Samplers::sampleGeometry(lightGeometry, dev_triangles, glm::vec3(random), lightNormal, lightSurfacePdf);

    if (lightSurfacePdf <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 hitPoint = incomingRay.getPositionAtTime(intersectionData.t);
    glm::vec3 directionToLight = lightPosition - hitPoint;
    float distance = glm::length(directionToLight);
    if (distance <= 0.0001f) {
        return glm::vec3(0.0f);
    }
	outDirectionToLight = directionToLight / distance;
	glm::vec3 L = directionToLight / distance;

	float cosThetaLight = glm::dot(lightNormal, -L);
	float cosThetaSurface = glm::dot(intersectionData.normal, L);

    if(cosThetaLight <= 0.0f || cosThetaSurface == 0.0f) {
        return glm::vec3(0.0f);
	}

    // Create visibility ray
    Ray visibilityRay = Ray::generateBouncedRay(intersectionData.normal, hitPoint, L);

    // Need to subtract by 2 * epsilon since we shift the ray origin by epsilon in the opposite direction
    if (!isVisible(visibilityRay, distance - (2.0f * RAY_OFFSET_EPSILON))) {
        return glm::vec3(0.0f);
    }

    float pArea = lightSurfacePdf * lightIndexPdf;
	outPdf = pArea * (distance * distance / cosThetaLight);

    return lightMaterial.emittance * lightMaterial.albedo;
}

__device__ bool DevScene::isVisible(const Ray& ray, float tMax)
{
    IntersectionData result = intersect(ray);
    return result.t < 0.0f || result.t > tMax;
}