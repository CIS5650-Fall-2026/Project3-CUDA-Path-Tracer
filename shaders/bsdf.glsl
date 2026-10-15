#ifndef BSDF_GLSL
#define BSDF_GLSL

#include "gpu/shared.h"
#include "common.glsl"
#include "rng.glsl"
#include "math.glsl"
#include "sampleWarping.glsl"

vec3 f_diffuse(vec3 albedo)
{
    return albedo * kInvPi;
}

vec3 sample_f_diffuse(vec3 albedo, vec2 xi, vec3 nor,
    out vec3 wiW, out float diffusePdf, out uint sampledType)
{
    // Make sure we set wiW to a world-space ray direction,
    // since wo is in tangent space.
    vec3 wiL = squareToHemisphereCosine(xi);
    diffusePdf = squareToHemisphereCosinePDF(wiL);
    wiW = localToWorld(nor) * wiL;
    sampledType = kDiffuseRefl;
    return f_diffuse(albedo);
}

vec3 sample_f_specular_refl(vec3 albedo, vec3 nor, vec3 wo,
    out vec3 wiW, out uint sampledType)
{
    // Make sure we set wiW to a world-space ray direction,
    // since wo is in tangent space
    vec3 wiL = vec3(-1.f * wo.x, -1.f * wo.y, wo.z);
    wiW = localToWorld(nor) * wiL;
    sampledType = kSpecRefl;
    return albedo / abs(wiL.z);
}

vec3 sample_f_specular_trans(vec3 albedo, float ior, vec3 nor, vec3 wo,
    out vec3 wiW, out uint sampledType)
{
    wiW = vec3(0.f);
    sampledType = kSpecTrans;

    const float etaA = 1.;

    // wo.z == dot(wo, vec3(0.f, 0.f, 1.f))
    bool isExitingSurface = wo.z < 0.f;

    vec3 refractNor = isExitingSurface ? vec3(0.f, 0.f, -1.f) : vec3(0.f, 0.f, 1.f);
    float relativeEta = isExitingSurface ? ior / etaA : etaA / ior;

    // Make sure we set wiW to a world-space ray direction,
    // since wo is in tangent space
    vec3 wiL = vec3(0.f);
    bool totalInternalReflection = !refractDir(normalize(wo), refractNor, relativeEta, wiL);

    if (totalInternalReflection)
    {
        return vec3(0.f);
    }

    wiW = localToWorld(nor) * wiL;
    sampledType = kSpecTrans;
    return albedo / abs(wiL.z);
}

vec3 fresnelDielectricEval(float cosThetaI, float ior)
{
    float etaI = 1.;
    cosThetaI = clamp(cosThetaI, -1.f, 1.f);

    bool entering = cosThetaI > 0.f;

    if (!entering)
    {
        float temp = etaI;
        etaI = ior;
        ior = temp;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / ior * sinThetaI;

    float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

    float rparl = ((ior * cosThetaI) - (etaI * cosThetaT)) / ((ior * cosThetaI) + (etaI * cosThetaT));
    float rperp = ((etaI * cosThetaI) - (ior * cosThetaT)) / ((etaI * cosThetaI) + (ior * cosThetaT));
    float eval = (rparl * rparl + rperp * rperp) / 2.f;

    return vec3(eval);
}

vec3 sample_f_glass(vec3 albedo, float ior, vec3 nor, vec3 wo,
    out vec3 wiW, out uint sampledType)
{
    float random = rng();
    if (random < 0.5)
    {
        // Have to double contribution b/c we only sample
        // reflection BxDF half the time
        vec3 R = sample_f_specular_refl(albedo, nor, wo, wiW, sampledType);
        sampledType = kSpecRefl;
        return 2 * fresnelDielectricEval(dot(nor, normalize(wiW)), ior) * R;
    }
    else
    {
        // Have to double contribution b/c we only sample
        // transmit BxDF half the time
        vec3 T = sample_f_specular_trans(albedo, ior, nor, wo, wiW, sampledType);
        sampledType = kSpecTrans;
        return 2 * (vec3(1.) - fresnelDielectricEval(dot(nor, normalize(wiW)), ior)) * T;
    }
}

vec3 computeAlbedo(SurfaceInteraction surface)
{
    vec3 albedo = surface.material.baseColor;
    return albedo;
}

float computeIor(SurfaceInteraction surface)
{
    return surface.material.ior;
}

vec3 computeNormal(SurfaceInteraction surface)
{
    vec3 nor = surface.isect.worldShadingNor;
    return nor;
}

// Computes the overall light scattering properties of a point on a Material,
// given the incoming and outgoing light directions.
vec3 f(SurfaceInteraction surface, vec3 woW, vec3 wiW)
{
    // Convert the incoming and outgoing light rays from
    // world space to local tangent space
    vec3 nor = computeNormal(surface);
    vec3 wo = worldToLocal(nor) * woW;
    vec3 wi = worldToLocal(nor) * wiW;

    // If the outgoing ray is parallel to the surface,
    // we know we can return black b/c the Lambert term
    // in the overall Light Transport Equation will be 0.
    if (wo.z == 0)
    {
        return vec3(0.f);
    }

    // Since GLSL does not support classes or polymorphism,
    // we have to handle each material type with its own function.
    if (surface.material.type == kDiffuseRefl)
    {
        return f_diffuse(computeAlbedo(surface));
    }

    // There is a 0% chance that a randomly chosen wi will be the perfect
    // reflection / refraction of wo, so any specular material will have
    // a BSDF of 0 when wi is chosen independently of the material.
    else if (surface.material.type == kSpecRefl
        || surface.material.type == kSpecTrans
        || surface.material.type == kSpecGlass)
    {
        return vec3(0.);
    }
    // Default case, unhandled material
    else
    {
        return vec3(1, 0, 1);
    }
}

// Sample_f() returns the same values as f(), but importantly it
// only takes in a wo. Note that wiW is declared as an "out vec3";
// this means the function is intended to compute and write a wi
// in world space (the trailing "W" indicates world space).
// In other words, Sample_f() evaluates the BSDF *after* generating
// a wi based on the SurfaceInteraction's material properties, allowing
// us to bias our wi samples in a way that gives more consistent
// light scattered along wo.
vec3 sample_f(SurfaceInteraction surface, vec3 woW, vec2 xi,
    out vec3 wiW, out float bsdfPdf, out uint sampledType)
{
    wiW = vec3(0.f);
    bsdfPdf = 0.f;
    sampledType = 0u;

    // Convert wo to local space from world space.
    // The various Sample_f()s output a wi in world space,
    // but assume wo is in local space.
    vec3 nor = computeNormal(surface);
    vec3 wo = worldToLocal(nor) * woW;

    if (surface.material.type == kDiffuseRefl)
    {
        return sample_f_diffuse(computeAlbedo(surface), xi, nor, wiW, bsdfPdf, sampledType);
    }
    else if (surface.material.type == kSpecRefl)
    {
        bsdfPdf = 1.;
        return sample_f_specular_refl(computeAlbedo(surface), nor, wo, wiW, sampledType);
    }
    else if (surface.material.type == kSpecTrans)
    {
        bsdfPdf = 1.;
        return sample_f_specular_trans(computeAlbedo(surface), computeIor(surface), nor, wo, wiW, sampledType);
    }
    else if (surface.material.type == kSpecGlass)
    {
        bsdfPdf = 1.;
        return sample_f_glass(computeAlbedo(surface), computeIor(surface), nor, wo, wiW, sampledType);
    }
    // Default case, unhandled material
    else
    {
        return vec3(1, 0, 1);
    }
}

// Compute the PDF of wi with respect to wo and the SurfaceInteraction's
// material properties.
float pdf(SurfaceInteraction surface, vec3 woW, vec3 wiW)
{
    vec3 nor = computeNormal(surface);
    vec3 wo = worldToLocal(nor) * woW;
    vec3 wi = worldToLocal(nor) * wiW;

    if (wo.z == 0)
    {
        // The cosine of this vector would be zero
        return 0.;
    }

    if (surface.material.type == kDiffuseRefl)
    {
        // Get the PDF of a Lambertian material
        return squareToHemisphereCosinePDF(wi);
    }
    else if (surface.material.type == kSpecRefl
        || surface.material.type == kSpecTrans
        || surface.material.type == kSpecGlass)
    {
        return 0.;
    }
    // Default case, unhandled material
    else
    {
        return 0.;
    }
}

float powerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / ((f * f) + (g * g));
}

float pdf_li(float dist, vec3 lightGeometricNor, vec3 wiW, float totalLightArea)
{
    float cosLight = absDot(lightGeometricNor, wiW);
    float denom = totalLightArea * cosLight;
    return (denom <= 0.0) ? 0.0 : ((dist * dist) / denom);
}

#endif