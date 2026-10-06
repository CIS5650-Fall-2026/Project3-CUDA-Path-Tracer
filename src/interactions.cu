#include "interactions.h"
#include "utilities.h"

#include <thrust/random.h>
#include <cmath>

// calculating ray direction: diffuse, reflection, refraction

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

__host__ __device__ glm::vec3 calculateReflectedRayDirection(
    glm::vec3 wo,
    glm::vec3 normal) {

    glm::vec3 wi = glm::reflect(wo, normal);

    return wi;
}

__host__ __device__ glm::vec3 calculateRefractedRayDirection(
    glm::vec3 wo,
    glm::vec3 normal,
    float ri) {

    float cosTheta = fminf(glm::dot(-wo, normal), 1.0f);
    glm::vec3 rPerp = ri * (wo + cosTheta * normal);
    glm::vec3 rParallel = -sqrt(fabsf(1.0f - (glm::length(rPerp) * glm::length(rPerp)))) * normal;

    return rPerp + rParallel;
}


// 2 implementations of schlick Fresnel approximation 

__host__ __device__ glm::vec3 metallicFresnel(
    glm::vec3 r0,
    float cosine) {
    float exponential = pow(1.0f - cosine, 5.0f);
    return r0 + (1.0f - r0) * exponential;
}

__host__ __device__ glm::vec3 diffuseFresnel(
    glm::vec3 r0,
    float r,
    float cosine) {
    float exponential = pow(1.0f - cosine, 5.0f);
    
    return r0 + (glm::max(glm::vec3(1.0f - r), r0) - r0) * exponential;
}

__host__ __device__ float transmissiveFresnel(
    float r0, 
    float cosine) {
    float exponential = pow(1.0f - cosine, 5.0f);
    return r0 + (1.0f - r0) * exponential;
}

// helpers for computing GGX distribution for Cook-Torrence specular lobe

__host__ __device__ glm::vec3 sphericalToCartesian(
    float theta,
    float phi
) {
    float x = sin(theta) * cos(phi);
    float y = cos(theta);
    float z = sin(theta) * sin(phi);

    return glm::vec3(x, y, z);
}

__host__ __device__ float clamp(
    float value,
    float min,
    float max
) {
    if (value < min) {
        return min;
    }
    else if (value > max) {
        return max;
    }

    return value;
}

// using GGX distribution, compute specular lobe 

__host__ __device__ float smithGGXMaskingShadowing(
    glm::vec3 wi,
    glm::vec3 wo,
    float a2) {

    // L = light/incoming direction aka wi
    // V = view/outgoing direction aka wo
    // N = surface normal aka (0, 1, 0) bc working in tangent space

    float dotNL = wi.y;
    float dotNV = wo.y;

    float denomA = dotNV * sqrt(a2 + (1.0f - a2) * dotNL * dotNL);
    float denomB = dotNL * sqrt(a2 + (1.0f - a2) * dotNV * dotNV);

    return 2.0f * dotNL * dotNV / (denomA + denomB);
}


__host__ __device__ void sampleGGXNorm(
    const Material &m,
    glm::vec3 wo, 
    glm::vec3 &wi,
    glm::vec3 &reflectance,
    thrust::default_random_engine &rng) {
    // theta = angle between microfacet normal & actual surface normal
    // lower roughness -> smaller theta, higher roughness -> larger theta
    // phi = direction geometric normal tilts from surface normal 

    // wi = direction of incoming ray in light transport equation 
    // want to choose direction wi
    // wo = outgoing ray direction 
    // wg = geometric normal from the surface

    float a2 = m.roughness * m.roughness;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float e0 = u01(rng);
    float e1 = u01(rng);

    float cosTheta = sqrt((1.0f - e0) / ((a2 - 1.0f) * e0 + 1.0f));
    cosTheta = clamp(cosTheta, 0.0f, 1.0f);

    float theta = acos(cosTheta);
    float phi = TWO_PI * e1;

    // microfacet normal 
    glm::vec3 wm = sphericalToCartesian(theta, phi);

    // calculate wi by reflecting wo about wm
    wi = 2.0f * glm::dot(wo, wm) * wm - wo;

    // want tangent space (normal is (0, 1, 0))
    // wi.y is the dot product w/ normal

    float wiwm = glm::dot(wi, wm);
    if (wi.y > 0.0f && wiwm > 0.0f) {

        // more reflected at grazing angles
        glm::vec3 F = metallicFresnel(m.color, wiwm);
        // less reflecting when there is shadowing/masking 
        float G = smithGGXMaskingShadowing(wi, wo, a2);

        float weight = abs(glm::dot(wo, wm)) / (wo.y * wm.y);

        // takes pdf into account 
        // calculating light comtribution of wi 
        // for diffuse lobe, estimator simplifies to just the albedo itself 
        reflectance = F * G * weight;
    }
    else {
        reflectance = glm::vec3(0.f);
    }
}

// scatter ray functions 

__host__ __device__ void scatterRayOpaque(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 texAlbedo,
    glm::vec3 texRough,
    Material &m,
    thrust::default_random_engine &rng)
{
 
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 albedo = m.color;
    if (m.texIdx != -1) {
        // sample
        albedo = texAlbedo;
    }

    if (m.roughmapIdx != -1) {
        // sample
        m.roughness = m.roughness * texRough[1];
        m.metallic = m.metallic * texRough[2];
    }

    // build frame around normal (tangent space) 
    glm::vec3 up = normal;
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
    glm::vec3 t = glm::normalize(glm::cross(up, directionNotNormal));
    glm::vec3 b = glm::normalize(glm::cross(up, t));

    // tangent space -> world
    glm::mat3 toWorld(t, up, b);

    // view direction in tangent space 
    glm::vec3 wo = glm::inverse(toWorld) * (-pathSegment.ray.direction);
    
    float cosTheta = glm::dot(normal, - pathSegment.ray.direction);
    glm::vec3 metallicF = metallicFresnel(albedo, cosTheta);
    glm::vec3 diffuseF = diffuseFresnel(albedo, m.roughness, cosTheta);
    // interpolate F based on m.metallic
    // bruh smth wrong with diffuseF
    glm::vec3 fresnel = (1.0f - m.metallic) * 0.04f + (m.metallic) * metallicF;
    // calculate luminance (in one value, how much light is reflected)
    float F = glm::dot(fresnel, glm::vec3(0.2126f, 0.7152f, 0.0722f)); // = ks for cook-torrence

    float p = u01(rng);
    // sample ray 
    glm::vec3 wi; 
    glm::vec3 reflectance;

    glm::vec3 wiWorld;
    if (p < F) { // sample specular lobe 
        sampleGGXNorm(m, wo, wi, reflectance, rng);
        wiWorld = toWorld * wi;
    }
    else { // sample diffuse lobe 
        wiWorld = calculateRandomDirectionInHemisphere(normal, rng);
        reflectance = albedo;
    }

    pathSegment.ray.origin = intersect + normal * EPSILON;
    pathSegment.ray.direction = wiWorld;
    pathSegment.color *= reflectance;
    pathSegment.remainingBounces--;
}

__host__ __device__ void scatterRayTransparent(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    bool outside,
    thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);

    bool frontFace = outside; // from intersection: did ray originate inside or outside object
    // assuming object in air (IOR = 1.0)
    // calculate ratio of IOR for Snell's law 
    float ri = frontFace ? (1 / m.refractionIndex) : m.refractionIndex;

    // assumes that normal is opposite to ray direction 
    float cosTheta = fminf(glm::dot(-pathSegment.ray.direction, normal), 1.0);

    // equation is broken - cannot refract 
    // total internal reflection (dense -> less dense & angle of incidence exceeds 
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    bool cannotRefract = ri * sinTheta > 1.0;

    // reflectivity varies with angle 
    // approximate fresnel w/ Schlick (similar to that used in GGX but with different base reflectance) 
    float r0 = pow((1 - ri) / (1 + ri), 2.0f);
    float reflectance = transmissiveFresnel(r0, cosTheta);

    // split reflection & refraction 
    glm::vec3 dir;
    //cannotRefract || reflectance > u01(rng)
    if (cannotRefract || reflectance > u01(rng)) {
        dir = calculateReflectedRayDirection(pathSegment.ray.direction, normal);
    }
    else {
        dir = calculateRefractedRayDirection(pathSegment.ray.direction, normal, ri);
    }

    // use Beer's law of absorption to calculate attenuation (tint the ray)
    // pure glass has attentuation = 1 (all channels survive bc no absorption) 
    glm::vec3 attenuation = glm::vec3(1.0f);

    //https://computergraphics.stackexchange.com/questions/297/is-this-the-correct-way-to-implement-beers-law
    // instead of checking that ray is leaving, check that its not entering to take acc of entire path!

    if (!outside) {
        // calculate distance traveled 
        float distTraveled = glm::length(intersect - pathSegment.ray.origin);
        attenuation *= glm::exp(-m.absorption * distTraveled);
    }

    pathSegment.color *= attenuation;

    // make sure that ray makes it out of medium! otherwise keep bouncing inside object -> color = white
    // suggestion from previous class works
    pathSegment.ray.origin = intersect + dir * (EPSILON * 500);
    pathSegment.ray.direction = dir;
    pathSegment.remainingBounces--;
}

__host__ __device__ void scatterRayFake(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    glm::vec3 dir = glm::vec3(0.f, 0.f, 0.f);
    if (m.hasReflective == 1.0f) {
        dir = calculateReflectedRayDirection(pathSegment.ray.direction, normal);
    }
    else {
        dir = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.origin = intersect + normal * EPSILON;
    pathSegment.ray.direction = dir;
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}