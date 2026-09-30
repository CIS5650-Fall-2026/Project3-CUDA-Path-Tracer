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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Nudge the new origin off the surface so the ray doesn't hit the same spot again
    const float rayOffset = 0.001f;
    glm::vec3 newDirection;

    if (m.hasRefractive > 0.0f)
    {
        // Glass: some light reflects, the rest bends through.
        // Note: the normal always faces the incoming ray.
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 incoming = pathSegment.ray.direction;

        // IOR ratio: air -> glass going in, glass -> air coming out
        float eta = outside ? (1.0f / m.indexOfRefraction) : m.indexOfRefraction;

        float cosIn = glm::clamp(glm::dot(-incoming, normal), 0.0f, 1.0f);
        float sinOutSquared = eta * eta * (1.0f - cosIn * cosIn);
        bool totalInternalReflection = sinOutSquared > 1.0f;

        // Schlick: how much light reflects (use the angle on the air side)
        float r0 = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
        r0 = r0 * r0;
        float cosForFresnel = cosIn;
        if (!outside && !totalInternalReflection)
        {
            cosForFresnel = sqrtf(1.0f - sinOutSquared);
        }
        float reflectChance = r0 + (1.0f - r0) * powf(1.0f - cosForFresnel, 5.0f);

        // Randomly pick one by the Fresnel odds (no extra weighting needed)
        glm::vec3 refracted = glm::refract(incoming, normal, eta);
        bool cannotRefract = totalInternalReflection || glm::dot(refracted, refracted) < 1e-8f;   // refract gives 0 on TIR
        if (cannotRefract || u01(rng) < reflectChance)
        {
            newDirection = glm::reflect(incoming, normal);
            pathSegment.ray.origin = intersect + normal * rayOffset;
        }
        else
        {
            newDirection = refracted;
            pathSegment.ray.origin = intersect - normal * rayOffset;   // start on the far side
        }
        pathSegment.color *= m.color;
        pathSegment.ray.direction = glm::normalize(newDirection);
        return;
    }

    if (m.hasReflective > 0.0f)
    {
        // Mirror: bounce straight off the surface, tinted by the specular color
        newDirection = glm::reflect(pathSegment.ray.direction, normal);

        // Rough metal: wiggle the mirror direction inside a Phong lobe
        // (GPU Gems 3 ch. 20, eq. 7-9). Bigger exponent = tighter lobe = shinier.
        if (m.specular.exponent > 0.0f)
        {
            thrust::uniform_real_distribution<float> u01(0, 1);
            float cosTheta = powf(u01(rng), 1.0f / (m.specular.exponent + 1.0f));
            float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
            float phi = TWO_PI * u01(rng);

            // Two directions perpendicular to the mirror direction, to aim around it
            glm::vec3 mirror = newDirection;
            glm::vec3 helper = (fabsf(mirror.x) < 0.9f) ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
            glm::vec3 side1 = glm::normalize(glm::cross(mirror, helper));
            glm::vec3 side2 = glm::cross(mirror, side1);
            newDirection = cosTheta * mirror + sinTheta * (cosf(phi) * side1 + sinf(phi) * side2);

            // Wiggled below the surface: nothing reflects that way, path done
            if (glm::dot(newDirection, normal) <= 0.0f)
            {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
                return;
            }
        }
        pathSegment.color *= m.specular.color;
    }
    else
    {
        // Diffuse: pick a random direction, more likely near the normal.
        // With this cosine-weighted sampling, the cosine term and the pdf cancel out,
        // so all that's left is multiplying by the base color.
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    pathSegment.ray.origin = intersect + normal * rayOffset;
    pathSegment.ray.direction = glm::normalize(newDirection);
}

__host__ __device__ void sampleLightSurface(
    const Geom& light,
    thrust::default_random_engine& rng,
    glm::vec3& point,
    glm::vec3& lightNormal,
    float& area)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (light.type == SPHERE)
    {
        // Uniform point on the unit sphere, then move it into place
        float z = 1.0f - 2.0f * u01(rng);
        float r = sqrtf(glm::max(0.0f, 1.0f - z * z));
        float phi = TWO_PI * u01(rng);
        glm::vec3 local(r * cosf(phi), r * sinf(phi), z);

        point = glm::vec3(light.transform * glm::vec4(0.5f * local, 1.0f));
        lightNormal = glm::normalize(glm::vec3(light.invTranspose * glm::vec4(local, 0.0f)));
        float radius = 0.5f * light.scale.x;
        area = 4.0f * PI * radius * radius;
        return;
    }

    // Cube: pick a face by its area, then select a point on it
    glm::vec3 axisX = glm::vec3(light.transform * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    glm::vec3 axisY = glm::vec3(light.transform * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
    glm::vec3 axisZ = glm::vec3(light.transform * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    float areaX = glm::length(glm::cross(axisY, axisZ));   // one face facing +/-x
    float areaY = glm::length(glm::cross(axisX, axisZ));
    float areaZ = glm::length(glm::cross(axisX, axisY));
    area = 2.0f * (areaX + areaY + areaZ);

    float pick = u01(rng) * (areaX + areaY + areaZ);
    int axis = 2;
    if (pick < areaX)
    {
        axis = 0;
    }
    else if (pick < areaX + areaY)
    {
        axis = 1;
    }
    float side = (u01(rng) < 0.5f) ? -0.5f : 0.5f;

    // Random spot on that face
    glm::vec3 local(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f);
    local[axis] = side;
    glm::vec3 localNormal(0.0f);
    localNormal[axis] = (side > 0.0f) ? 1.0f : -1.0f;

    point = glm::vec3(light.transform * glm::vec4(local, 1.0f));
    lightNormal = glm::normalize(glm::vec3(light.invTranspose * glm::vec4(localNormal, 0.0f)));
}

__host__ __device__ void sampleDirectLight(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    const Geom* geoms,
    const int* lightIndices,
    int numLights,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Pick one light, then a point on it
    int pickedLight = glm::min((int)(u01(rng) * numLights), numLights - 1);
    const Geom& light = geoms[lightIndices[pickedLight]];

    glm::vec3 lightPoint;
    glm::vec3 lightNormal;
    float lightArea;
    sampleLightSurface(light, rng, lightPoint, lightNormal, lightArea);
    lightPoint += light.velocity * pathSegment.time;   // where a moving light is at this moment

    glm::vec3 toLight = lightPoint - intersect;
    float distSquared = glm::dot(toLight, toLight);
    glm::vec3 direction = toLight / sqrtf(distSquared);
    float cosSurface = glm::dot(direction, normal);
    float cosLight = glm::dot(-direction, lightNormal);

    // Point is behind us, or on a light face pointing away: no light, path done
    if (cosSurface <= 0.0f || cosLight <= 0.0f)
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    // (color / pi) * cos / pdf, with pdf = dist^2 / (cosLight * area * numLights)
    // Light is added when the ray actually hits it (blocked = black)
    pathSegment.color *= m.color * cosSurface * cosLight * lightArea * (float)numLights / (PI * distSquared);

    const float rayOffset = 0.001f;
    pathSegment.ray.origin = intersect + normal * rayOffset;
    pathSegment.ray.direction = direction;
}
