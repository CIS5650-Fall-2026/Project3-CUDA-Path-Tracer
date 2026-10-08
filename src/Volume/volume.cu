#include "volume.h"
#include "../utilities.h"
#include "../cuda-utilities.h"

__host__ __device__ float henyeyGreensteinSingle(float cosTheta, float gPure, VolumeData& vd) {
     float gMulti = gPure * powf(vd.gDecay, static_cast<float>(vd.bounceDepth));

    float denom = 1.0f + gMulti*gMulti + 2.0f * gMulti * cosTheta;
    return 0.25 * IPI * (1.0f - gMulti*gMulti) / (denom * max(1e-5, sqrt(denom)));
}

__host__ __device__ float henyeyGreensteinDouble(float cosTheta, float g1, float g2, float blend, VolumeData& vd) {
    return henyeyGreensteinSingle(cosTheta, g1, vd) * (1.0f - blend) + henyeyGreensteinSingle(cosTheta, g2, vd) * blend;
}

__host__ __device__ glm::vec3 sampleHenyeyGreensteinSingle(glm::vec3 wo, float gPure, VolumeData& vd, thrust::default_random_engine& rng, float& outPdf) {
      thrust::uniform_real_distribution u01(0.0f, 1.0f);
      glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
      float gMulti = gPure * powf(vd.gDecay, static_cast<float>(vd.bounceDepth));

        float cosTheta;
        if (glm::abs(gMulti) < 1e-3f)
            cosTheta = 1.0f - 2.0f * xi.x;
        else
            cosTheta = -1.0f / (2.0f * gMulti) *
                        (1.0f + SQR(gMulti) - SQR((1 - SQR(gMulti)) / (1.0f + gMulti - 2.0f * gMulti * xi.x)));

        float sinTheta = sqrt(max(1 - SQR(cosTheta), 1e-5));
        float phi = 2.0f * PI * xi.y;
        glm::mat3 wFrame = FrameFromZ(wo);
        glm::vec3 wi = wFrame * (SphericalDirection(cosTheta, sinTheta, phi));

       outPdf = henyeyGreensteinSingle(cosTheta, gPure, vd);
       return wi;
}

__host__ __device__ glm::vec3 sampleHenyeyGreensteinDouble(glm::vec3 wo, float g1, float g2, float blend, VolumeData& vd, thrust::default_random_engine& rng, float& outPdf) {
    thrust::uniform_real_distribution u01(0.0f, 1.0f);

    bool gChoice = u01(rng) <= blend;
    float g = gChoice ? g2 : g1;
    float pdf = 100.0f;
    glm::vec3 wi = sampleHenyeyGreensteinSingle(wo, g, vd, rng, pdf);

    outPdf = pdf * (gChoice ? blend : 1.0f - blend);

    return wi;
}


__host__ __device__ float sampleVolume(const Volume& volume, glm::vec3 p) {
    if (length(p) > 3.0f) return 0.0f;
    return glm::smoothstep(0.75f, 1.0f, glm::abs(glm::dot(cos(2.0f * p), sin(4.0f * glm::vec3(p.z, p.y, p.x))))) * 2.0f;// / max(0.5f, dot(p, p));
    // // Weird way to do this, makes more sense physically to do absorption (a), scattering (s), then extinction (e) = a + s, albedo = s/e, absorption = a/e
    // float extinction = 10.0f / max(0.5f, dot(p, p));
    
    // float albedo = 0.99f;
    

    // float absorption = (1.0f - albedo) * extinction;

    // return glm::vec3(extinction, albedo, absorption);
}

__host__ __device__ float volumeIntersectionTest(Ray ray, const Volume& volume, thrust::default_random_engine& rng, float tVolumeStart, float tMax) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float t = tVolumeStart;
    int iter = 0;
    while(t < tMax) {
        t += -log(max(1e-5, u01(rng))) / volume.extinctionMult;

        if(t >= tMax) {
            return -1.0f;
        }

        glm::vec3 newPoint = ray.origin + ray.direction * t;
        float extinction = sampleVolume(volume, newPoint);

        if(u01(rng) <= extinction / volume.extinctionMult) {
            return t;
        }

        // Iter cap
        if (iter++ > 500)
            return t;
    }

    return -1.0f;
}