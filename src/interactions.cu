#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 calculateSampledDirectionInHemisphere(glm::vec3 normal, float sample1, float sample2)
{
    float up = sqrtf(sample1);
    float over = sqrtf(1.0f - up * up);
    float around = sample2 * TWO_PI;

    glm::vec3 directionNotNormal;

    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal + cosf(around) * over * perpendicularDirection1 + sinf(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    bool outside,
    bool enableRefraction,
    bool useLowDiscrepancy,
    int iter,
    int depth,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    pathSegment.color *= m.color;

    if (enableRefraction && m.hasRefractive > 0.0f)
    {
        glm::vec3 incomingDirection = glm::normalize(pathSegment.ray.direction);
        glm::vec3 surfaceNormal = glm::normalize(normal);

        float n1;
        float n2;

        if (outside)
        {
            n1 = 1.0f;
            n2 = m.indexOfRefraction;
        }
        else
        {
            n1 = m.indexOfRefraction;
            n2 = 1.0f;
        }

        float eta = n1 / n2;

        float cosTheta = glm::clamp(
            glm::dot(-incomingDirection, surfaceNormal),
            0.0f,
            1.0f
        );

        float sinThetaSquared = eta * eta * (1.0f - cosTheta * cosTheta);

        bool totalInternalReflection = sinThetaSquared > 1.0f;

        float r0 = (n1 - n2) / (n1 + n2);

        r0 = r0 * r0;

        float oneMinusCos = 1.0f - cosTheta;

        float oneMinusCosSquared = oneMinusCos * oneMinusCos;

        float oneMinusCosFifth = oneMinusCosSquared * oneMinusCosSquared * oneMinusCos;

        float reflectProbability = r0 + (1.0f - r0) * oneMinusCosFifth;

        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        float randomSample = u01(rng);

        glm::vec3 newDirection;

        if (totalInternalReflection ||
            randomSample < reflectProbability)
        {
            newDirection =
                glm::reflect(
                    incomingDirection,
                    surfaceNormal
                );
        }
        else
        {
            newDirection =
                glm::refract(
                    incomingDirection,
                    surfaceNormal,
                    eta
                );
        }

        newDirection = glm::normalize(newDirection);

        pathSegment.ray.origin = intersect + newDirection * 0.001f;

        pathSegment.ray.direction = newDirection;

        pathSegment.allowEmission = true;
    }
    else
    {
        pathSegment.ray.origin = intersect;

        if (useLowDiscrepancy)
        {
            unsigned int sampleIndex = (unsigned int)(iter + 1);
            glm::vec2 sample = halton2D(sampleIndex, 2, 3);

            unsigned int seed = (unsigned int)pathSegment.pixelIndex ^ ((unsigned int)depth * 0x9e3779b9u);

            sample.x += hashToUnitFloat(seed);
            sample.y += hashToUnitFloat(seed ^ 0x68bc21ebu);

            sample.x -= floorf(sample.x);
            sample.y -= floorf(sample.y);

            pathSegment.ray.direction = calculateSampledDirectionInHemisphere(normal, sample.x, sample.y);
        }
        else
        {
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        }
        pathSegment.allowEmission = false;
    }
}

__device__ bool rayAABBIntersection(const Ray& ray, const glm::vec3& minBounds, const glm::vec3& maxBounds, float maxT)
{
    float tMin = 0.0f;
    float tMax = maxT;

    for (int axis = 0; axis < 3; axis++)
    {
        float origin = ray.origin[axis];
        float direction = ray.direction[axis];

        if (fabsf(direction) < 0.000001f)
        {
            if (origin < minBounds[axis] || origin > maxBounds[axis])
            {
                return false;
            }

            continue;
        }

        float invDirection = 1.0f / direction;
        float t0 = (minBounds[axis] - origin) * invDirection;
        float t1 = (maxBounds[axis] - origin) * invDirection;

        if (t0 > t1)
        {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);

        if (tMax < tMin)
        {
            return false;
        }
    }

    return true;
}