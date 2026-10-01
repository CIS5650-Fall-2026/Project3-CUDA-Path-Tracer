#include "interactions.h"

#include "intersections.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ float nextSample(Sampler& sampler)
{
    const int primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131 };
    const int primeCount = 32;

    int dimension = sampler.dimension++;
    if (!sampler.halton || dimension >= primeCount)
    {
        thrust::uniform_real_distribution<float> u01(0, 1);
        return u01(sampler.rng);
    }

    int base = primes[dimension];
    float inverseBase = 1.0f / base;
    float fraction = inverseBase;
    float value = 0.0f;
    for (int i = sampler.index; i > 0; i /= base)
    {
        value += (i % base) * fraction;
        fraction *= inverseBase;
    }

    value += (utilhash(sampler.pixel * primeCount + dimension) & 0xFFFFFF) / 16777216.0f;
    return glm::min(value - floorf(value), 0.999999f);
}

__host__ __device__ glm::vec3 calculateRandomDirectionInSphere(Sampler &sampler)
{
    float z = 1.0f - 2.0f * nextSample(sampler);
    float radius = sqrt(1.0f - z * z);
    float angle = TWO_PI * nextSample(sampler);
    return glm::vec3(radius * cos(angle), radius * sin(angle), z);
}

__host__ __device__ float schlickReflectance(float cosTheta, float indexOfRefraction)
{
    float r0 = (1.0f - indexOfRefraction) / (1.0f + indexOfRefraction);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    Sampler &sampler)
{
    float up = sqrt(nextSample(sampler)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = nextSample(sampler) * TWO_PI;

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
    Sampler &sampler)
{
    glm::vec3 offset = normal * 0.001f;
    if (m.scatterDistance > 0.0f)
    {
        float cosIncident = glm::dot(-pathSegment.ray.direction, normal);
        if (outside && nextSample(sampler) < schlickReflectance(cosIncident, m.indexOfRefraction))
        {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else
        {
            pathSegment.mediumDistance = outside ? m.scatterDistance : 0.0f;
            pathSegment.mediumColor = m.color;
            offset = -offset;
        }
    }
    else if (m.hasRefractive > 0.0f)
    {
        glm::vec3 incident = pathSegment.ray.direction;
        float eta = outside ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
        float cosIncident = glm::dot(-incident, normal);
        float sin2Transmitted = eta * eta * (1.0f - cosIncident * cosIncident);

        float reflectance = 1.0f;
        if (sin2Transmitted < 1.0f)
        {
            float cosTheta = outside ? cosIncident : sqrt(1.0f - sin2Transmitted);
            reflectance = schlickReflectance(cosTheta, m.indexOfRefraction);
        }

        if (nextSample(sampler) < reflectance)
        {
            pathSegment.ray.direction = glm::reflect(incident, normal);
        }
        else
        {
            pathSegment.ray.direction = glm::refract(incident, normal, eta);
            offset = -offset;
        }
        pathSegment.color *= m.specular.color;
    }
    else if (m.hasReflective > 0.0f)
    {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
    }
    else
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, sampler);
        pathSegment.color *= m.color;
    }
    pathSegment.ray.origin = intersect + offset;
}

__host__ __device__ float valueNoise(glm::vec3 p)
{
    glm::vec3 cell = glm::floor(p);
    glm::vec3 f = p - cell;
    f = f * f * (3.0f - 2.0f * f);

    float result = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        glm::vec3 corner(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        glm::ivec3 c(cell + corner);
        float value = (utilhash(c.x * 73856093 ^ c.y * 19349663 ^ c.z * 83492791) & 0xFFFF) / 65535.0f;
        glm::vec3 weight = corner * f + (1.0f - corner) * (1.0f - f);
        result += value * weight.x * weight.y * weight.z;
    }
    return result;
}

__host__ __device__ float proceduralTexture(const Material &m, glm::vec3 point)
{
    glm::vec3 p = point * m.textureScale;
    if (m.texture == CHECKER)
    {
        int cell = (int)(floor(p.x) + floor(p.y) + floor(p.z));
        return (cell & 1) ? 1.0f : 0.2f;
    }

    float turbulence = 0.0f;
    float amplitude = 1.0f;
    glm::vec3 q = p;
    for (int octave = 0; octave < 4; octave++)
    {
        turbulence += amplitude * valueNoise(q);
        q *= 2.0f;
        amplitude *= 0.5f;
    }
    return 0.6f + 0.4f * sin((p.x + p.y + p.z) * 2.0f + 5.0f * turbulence);
}

__host__ __device__ glm::vec3 texelAt(const Texture &texture, const glm::vec3 *texels, int x, int y)
{
    x = (x % texture.width + texture.width) % texture.width;
    y = (y % texture.height + texture.height) % texture.height;
    return texels[texture.offset + y * texture.width + x];
}

__host__ __device__ glm::vec3 sampleTexture(const Texture &texture, const glm::vec3 *texels, glm::vec2 uv)
{
    float x = uv.x * texture.width - 0.5f;
    float y = uv.y * texture.height - 0.5f;
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    float fx = x - x0;
    float fy = y - y0;
    return glm::mix(
        glm::mix(texelAt(texture, texels, x0, y0), texelAt(texture, texels, x0 + 1, y0), fx),
        glm::mix(texelAt(texture, texels, x0, y0 + 1), texelAt(texture, texels, x0 + 1, y0 + 1), fx),
        fy);
}

__host__ __device__ glm::vec3 bumpNormal(
    const Material &m,
    const Texture &texture,
    const glm::vec3 *texels,
    glm::vec2 uv,
    glm::vec3 normal,
    glm::vec3 tangent)
{
    glm::vec3 t = tangent - normal * glm::dot(normal, tangent);
    if (glm::dot(t, t) < EPSILON)
    {
        return normal;
    }
    t = glm::normalize(t);
    glm::vec3 b = glm::cross(normal, t);

    if (m.normalMap)
    {
        glm::vec3 n = sampleTexture(texture, texels, uv) * 2.0f - 1.0f;
        return glm::normalize(t * n.x + b * n.y + normal * n.z);
    }

    float height = sampleTexture(texture, texels, uv).r;
    float du = sampleTexture(texture, texels, uv + glm::vec2(1.0f / texture.width, 0.0f)).r - height;
    float dv = sampleTexture(texture, texels, uv + glm::vec2(0.0f, 1.0f / texture.height)).r - height;
    return glm::normalize(normal - m.bumpScale * (du * t - dv * b));
}

__host__ __device__ glm::vec3 samplePointOnLight(
    const Geom &light,
    Sampler &sampler,
    glm::vec3 &lightNormal,
    float &area)
{
    glm::vec3 point;
    glm::vec3 normal(0.0f);
    if (light.type == CUBE)
    {
        glm::vec3 faceAreas(
            light.scale.y * light.scale.z,
            light.scale.x * light.scale.z,
            light.scale.x * light.scale.y);
        float halfArea = faceAreas.x + faceAreas.y + faceAreas.z;
        float pick = nextSample(sampler) * halfArea;
        int axis = pick < faceAreas.x ? 0 : (pick < faceAreas.x + faceAreas.y ? 1 : 2);

        point[axis] = nextSample(sampler) < 0.5f ? -0.5f : 0.5f;
        point[(axis + 1) % 3] = nextSample(sampler) - 0.5f;
        point[(axis + 2) % 3] = nextSample(sampler) - 0.5f;
        normal[axis] = point[axis];
        area = 2.0f * halfArea;
    }
    else
    {
        normal = calculateRandomDirectionInSphere(sampler);
        point = normal * 0.5f;
        area = PI * light.scale.x * light.scale.x;
    }

    lightNormal = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(normal, 0.0f)));
    return multiplyMV(light.transform, glm::vec4(point, 1.0f));
}

__host__ __device__ void scatterRayToLight(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    const Geom *geoms,
    const int *lights,
    int lightCount,
    Sampler &sampler)
{
    int lightIndex = glm::min((int)(nextSample(sampler) * lightCount), lightCount - 1);
    const Geom &light = geoms[lights[lightIndex]];

    glm::vec3 lightNormal;
    float area;
    glm::vec3 lightPoint = samplePointOnLight(light, sampler, lightNormal, area)
        + light.velocity * pathSegment.time;

    glm::vec3 toLight = lightPoint - intersect;
    float distance2 = glm::dot(toLight, toLight);
    glm::vec3 direction = toLight / sqrt(distance2);
    float cosSurface = glm::max(glm::dot(normal, direction), 0.0f);
    float cosLight = glm::max(glm::dot(lightNormal, -direction), 0.0f);

    float weight = cosSurface * cosLight * area * lightCount / (PI * distance2);
    pathSegment.color *= m.color * glm::min(weight, 1.0f);
    pathSegment.ray.origin = intersect + normal * 0.001f;
    pathSegment.ray.direction = direction;
}
