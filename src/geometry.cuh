#pragma once

#include "geometry.h"

#include <cfloat>
#include <climits>
#include <cmath>

#if defined(__CUDACC__)
#define GEOMETRY_HD __host__ __device__
#define GEOMETRY_INLINE __forceinline__
#define GEOMETRY_NOINLINE __noinline__
#else
#define GEOMETRY_HD
#define GEOMETRY_INLINE inline
#define GEOMETRY_NOINLINE inline
#endif

namespace geometry_detail
{

GEOMETRY_HD GEOMETRY_INLINE float absolute(float value)
{
    return value < 0.0f ? -value : value;
}

GEOMETRY_HD GEOMETRY_INLINE float maxAbsComponent(const glm::vec3& value)
{
    float result = absolute(value.x);
    if (absolute(value.y) > result)
    {
        result = absolute(value.y);
    }
    if (absolute(value.z) > result)
    {
        result = absolute(value.z);
    }
    return result;
}

GEOMETRY_HD GEOMETRY_INLINE bool finiteFloat(float value)
{
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

GEOMETRY_HD GEOMETRY_INLINE bool finiteDouble(double value)
{
    return value == value && value <= DBL_MAX && value >= -DBL_MAX;
}

GEOMETRY_HD GEOMETRY_INLINE double dotDouble(const glm::vec3& a,
                                              const glm::vec3& b)
{
    return static_cast<double>(a.x) * static_cast<double>(b.x) +
           static_cast<double>(a.y) * static_cast<double>(b.y) +
           static_cast<double>(a.z) * static_cast<double>(b.z);
}

GEOMETRY_HD GEOMETRY_INLINE bool finiteVec2(const glm::vec2& value)
{
    return finiteFloat(value.x) && finiteFloat(value.y);
}

GEOMETRY_HD GEOMETRY_INLINE bool finiteVec3(const glm::vec3& value)
{
    return finiteFloat(value.x) && finiteFloat(value.y) && finiteFloat(value.z);
}

GEOMETRY_HD GEOMETRY_INLINE float clamp01(float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

GEOMETRY_HD GEOMETRY_INLINE float clampSignedUnit(float value)
{
    return value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
}

GEOMETRY_HD GEOMETRY_INLINE glm::vec3 transformPoint(const glm::mat4& matrix,
                                                      const glm::vec3& point)
{
    return glm::vec3(matrix * glm::vec4(point, 1.0f));
}

GEOMETRY_HD GEOMETRY_INLINE glm::vec3 transformVector(const glm::mat4& matrix,
                                                       const glm::vec3& vector)
{
    return glm::vec3(matrix * glm::vec4(vector, 0.0f));
}

GEOMETRY_HD GEOMETRY_INLINE bool safeNormalize(const glm::vec3& vector,
                                                glm::vec3& normalized)
{
    if (!finiteVec3(vector))
    {
        return false;
    }
    const float scale = maxAbsComponent(vector);
    if (!(scale > 0.0f))
    {
        return false;
    }
    const glm::vec3 scaled = vector / scale;
    const float squaredLength = glm::dot(scaled, scaled);
    if (!finiteFloat(squaredLength) || !(squaredLength > 0.0f))
    {
        return false;
    }
    normalized = scaled * (1.0f / sqrtf(squaredLength));
    return finiteVec3(normalized);
}

GEOMETRY_HD GEOMETRY_INLINE void tangentFrame(const glm::vec3& normal,
                                               glm::vec3& dpdu,
                                               glm::vec3& dpdv)
{
    const glm::vec3 helper = absolute(normal.z) < 0.999f
                                 ? glm::vec3(0.0f, 0.0f, 1.0f)
                                 : glm::vec3(0.0f, 1.0f, 0.0f);
    if (!safeNormalize(glm::cross(helper, normal), dpdu))
    {
        dpdu = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    dpdv = glm::cross(normal, dpdu);
}

GEOMETRY_HD GEOMETRY_INLINE void validateTangents(const glm::vec3& normal,
                                                   glm::vec3& dpdu,
                                                   glm::vec3& dpdv)
{
    glm::vec3 unitU;
    glm::vec3 unitV;
    if (!safeNormalize(dpdu, unitU) || !safeNormalize(dpdv, unitV))
    {
        tangentFrame(normal, dpdu, dpdv);
        return;
    }
    const glm::vec3 tangentCross = glm::cross(unitU, unitV);
    if (!(glm::dot(tangentCross, tangentCross) > 1.0e-20f))
    {
        tangentFrame(normal, dpdu, dpdv);
    }
}

GEOMETRY_HD GEOMETRY_INLINE bool validRay(const Ray& ray)
{
    if (!finiteVec3(ray.origin) || !finiteVec3(ray.direction))
    {
        return false;
    }
    return maxAbsComponent(ray.direction) > 0.0f;
}

GEOMETRY_HD GEOMETRY_INLINE void finishHit(const Geom& geom,
                                            int primitiveId,
                                            const Ray& ray,
                                            float t,
                                            const glm::vec3& position,
                                            const glm::vec3& geometricNormal,
                                            const glm::vec3& shadingNormal,
                                            const glm::vec2& uv,
                                            const glm::vec3& dpdu,
                                            const glm::vec3& dpdv,
                                            Hit& hit)
{
    hit.t = t;
    hit.primitiveId = primitiveId;
    hit.materialId = geom.materialid;
    hit.position = position;
    hit.geometricNormal = geometricNormal;
    hit.shadingNormal = shadingNormal;
    hit.dpdu = dpdu;
    hit.dpdv = dpdv;
    hit.uv = uv;
    const float directionScale = maxAbsComponent(ray.direction);
    hit.frontFace = directionScale > 0.0f &&
                    glm::dot(ray.direction / directionScale, geometricNormal) < 0.0f;
}

struct TriangleParameters
{
    float t;
    float bary1;
    float bary2;
};

GEOMETRY_HD GEOMETRY_INLINE bool solveTriangleDouble(const Geom& geom,
                                                      const Ray& ray,
                                                      const glm::vec3& scaledDirection,
                                                      float directionScale,
                                                      float tMin,
                                                      float tMax,
                                                      TriangleParameters& result)
{
    const double e1x = static_cast<double>(geom.vertices[1].x) - geom.vertices[0].x;
    const double e1y = static_cast<double>(geom.vertices[1].y) - geom.vertices[0].y;
    const double e1z = static_cast<double>(geom.vertices[1].z) - geom.vertices[0].z;
    const double e2x = static_cast<double>(geom.vertices[2].x) - geom.vertices[0].x;
    const double e2y = static_cast<double>(geom.vertices[2].y) - geom.vertices[0].y;
    const double e2z = static_cast<double>(geom.vertices[2].z) - geom.vertices[0].z;
    const double dx = scaledDirection.x;
    const double dy = scaledDirection.y;
    const double dz = scaledDirection.z;

    const double px = dy * e2z - dz * e2y;
    const double py = dz * e2x - dx * e2z;
    const double pz = dx * e2y - dy * e2x;
    const double determinant = e1x * px + e1y * py + e1z * pz;
    const double edge1Length2 = e1x * e1x + e1y * e1y + e1z * e1z;
    const double edge2Length2 = e2x * e2x + e2y * e2y + e2z * e2z;
    const double directionLength2 = dx * dx + dy * dy + dz * dz;
    const double determinantScale2 = edge1Length2 * edge2Length2 * directionLength2;
    if (!finiteDouble(determinant) || !finiteDouble(determinantScale2) ||
        !(determinantScale2 > 0.0) ||
        absolute(static_cast<float>(determinant / sqrt(determinantScale2))) <=
            8.0f * FLT_EPSILON)
    {
        return false;
    }

    const double fx = static_cast<double>(ray.origin.x) - geom.vertices[0].x;
    const double fy = static_cast<double>(ray.origin.y) - geom.vertices[0].y;
    const double fz = static_cast<double>(ray.origin.z) - geom.vertices[0].z;
    const double inverseDeterminant = 1.0 / determinant;
    const double bary1 = (fx * px + fy * py + fz * pz) * inverseDeterminant;
    if (!finiteDouble(bary1) || bary1 < 0.0 || bary1 > 1.0)
    {
        return false;
    }

    const double qx = fy * e1z - fz * e1y;
    const double qy = fz * e1x - fx * e1z;
    const double qz = fx * e1y - fy * e1x;
    const double bary2 = (dx * qx + dy * qy + dz * qz) * inverseDeterminant;
    if (!finiteDouble(bary2) || bary2 < 0.0 || bary1 + bary2 > 1.0)
    {
        return false;
    }

    const double scaledT = (e2x * qx + e2y * qy + e2z * qz) * inverseDeterminant;
    const double tDouble = scaledT / static_cast<double>(directionScale);
    if (!finiteDouble(tDouble) || tDouble < -static_cast<double>(FLT_MAX) ||
        tDouble > static_cast<double>(FLT_MAX))
    {
        return false;
    }
    const float t = static_cast<float>(tDouble);
    if (t < tMin || t > tMax)
    {
        return false;
    }
    result.t = t;
    result.bary1 = static_cast<float>(bary1);
    result.bary2 = static_cast<float>(bary2);
    return true;
}

GEOMETRY_HD GEOMETRY_INLINE bool intersectTriangle(const Geom& geom,
                                                    int primitiveId,
                                                    const Ray& ray,
                                                    float tMin,
                                                    float tMax,
                                                    Hit& hit)
{
    const glm::vec3 edge1 = geom.vertices[1] - geom.vertices[0];
    const glm::vec3 edge2 = geom.vertices[2] - geom.vertices[0];
    const glm::vec3 normalVector = glm::cross(edge1, edge2);
    glm::vec3 geometricNormal;
    if (!safeNormalize(normalVector, geometricNormal))
    {
        return false;
    }

    const float directionScale = maxAbsComponent(ray.direction);
    if (!(directionScale > 0.0f))
    {
        return false;
    }
    const glm::vec3 scaledDirection = ray.direction / directionScale;
    const glm::vec3 p = glm::cross(scaledDirection, edge2);
    const float determinant = glm::dot(edge1, p);
    const float determinantScale2 = glm::dot(edge1, edge1) *
                                    glm::dot(edge2, edge2) *
                                    glm::dot(scaledDirection, scaledDirection);
    TriangleParameters parameters{};
    bool useDouble = !finiteFloat(determinant) || !finiteFloat(determinantScale2) ||
                     !(determinantScale2 > 0.0f) ||
                     absolute(determinant) <=
                         8.0f * FLT_EPSILON * sqrtf(determinantScale2);
    const glm::vec3 fromVertex = ray.origin - geom.vertices[0];
    if (!useDouble)
    {
        const float inverseDeterminant = 1.0f / determinant;
        const float bary1 = glm::dot(fromVertex, p) * inverseDeterminant;
        const glm::vec3 q = glm::cross(fromVertex, edge1);
        const float bary2 = glm::dot(scaledDirection, q) * inverseDeterminant;
        const float edgeTolerance = 16.0f * FLT_EPSILON;
        if (!finiteFloat(bary1) || !finiteFloat(bary2))
        {
            useDouble = true;
        }
        else if (bary1 < 0.0f || bary1 > 1.0f || bary2 < 0.0f ||
                 bary1 + bary2 > 1.0f)
        {
            if (bary1 < -edgeTolerance || bary1 > 1.0f + edgeTolerance ||
                bary2 < -edgeTolerance || bary1 + bary2 > 1.0f + edgeTolerance)
            {
                return false;
            }
            useDouble = true;
        }
        else
        {
            const float t = (glm::dot(edge2, q) * inverseDeterminant) / directionScale;
            if (!finiteFloat(t))
            {
                useDouble = true;
            }
            else if (t < tMin || t > tMax)
            {
                return false;
            }
            else
            {
                parameters.t = t;
                parameters.bary1 = bary1;
                parameters.bary2 = bary2;
            }
        }
    }
    if (useDouble && !solveTriangleDouble(geom, ray, scaledDirection, directionScale,
                                          tMin, tMax, parameters))
    {
        return false;
    }

    const float t = parameters.t;
    const float bary1 = parameters.bary1;
    const float bary2 = parameters.bary2;
    const float bary0 = 1.0f - bary1 - bary2;
    glm::vec2 uv0(0.0f, 0.0f);
    glm::vec2 uv1(1.0f, 0.0f);
    glm::vec2 uv2(0.0f, 1.0f);
    if (geom.hasUVs)
    {
        uv0 = geom.uvs[0];
        uv1 = geom.uvs[1];
        uv2 = geom.uvs[2];
        if (!finiteVec2(uv0) || !finiteVec2(uv1) || !finiteVec2(uv2))
        {
            return false;
        }
    }
    const glm::vec2 uv = bary0 * uv0 + bary1 * uv1 + bary2 * uv2;

    glm::vec3 shadingNormal = geometricNormal;
    if (geom.hasNormals)
    {
        const glm::vec3 interpolated = bary0 * geom.normals[0] +
                                       bary1 * geom.normals[1] +
                                       bary2 * geom.normals[2];
        if (safeNormalize(interpolated, shadingNormal))
        {
            if (glm::dot(shadingNormal, geometricNormal) < 0.0f)
            {
                shadingNormal = -shadingNormal;
            }
        }
        else
        {
            shadingNormal = geometricNormal;
        }
    }

    glm::vec3 dpdu;
    glm::vec3 dpdv;
    if (!geom.hasUVs)
    {
        // These are the derivatives of the stable fallback UV assignment above.
        dpdu = edge1;
        dpdv = edge2;
    }
    else
    {
        const glm::vec2 delta1 = uv1 - uv0;
        const glm::vec2 delta2 = uv2 - uv0;
        const float denominator = delta1.x * delta2.y - delta1.y * delta2.x;
        if (finiteFloat(denominator) && absolute(denominator) > 1.0e-12f)
        {
            const float inverse = 1.0f / denominator;
            dpdu = (edge1 * delta2.y - edge2 * delta1.y) * inverse;
            dpdv = (edge2 * delta1.x - edge1 * delta2.x) * inverse;
        }
        else
        {
            tangentFrame(geometricNormal, dpdu, dpdv);
        }
    }
    validateTangents(geometricNormal, dpdu, dpdv);

    const glm::vec3 position = ray.origin + t * ray.direction;
    if (!finiteVec3(position) || !finiteVec2(uv))
    {
        return false;
    }
    finishHit(geom, primitiveId, ray, t, position, geometricNormal,
              shadingNormal, uv, dpdu, dpdv, hit);
    return true;
}

GEOMETRY_HD GEOMETRY_INLINE bool intersectSphere(const Geom& geom,
                                                  int primitiveId,
                                                  const Ray& ray,
                                                  float tMin,
                                                  float tMax,
                                                  Hit& hit)
{
    const glm::vec3 localOrigin = transformPoint(geom.inverseTransform, ray.origin);
    const glm::vec3 localDirection = transformVector(geom.inverseTransform, ray.direction);
    if (!finiteVec3(localOrigin) || !finiteVec3(localDirection))
    {
        return false;
    }

    float t = 0.0f;
    glm::vec3 localPosition(0.0f);
    bool useDouble = true;

    // Ordinary rendering rays take an all-float q-form path. Only coefficient
    // overflow/underflow or discriminant cancellation routes to FP64.
    const float aFloat = glm::dot(localDirection, localDirection);
    const float halfBFloat = glm::dot(localOrigin, localDirection);
    const float cFloat = glm::dot(localOrigin, localOrigin) - 0.25f;
    const float halfBSquared = halfBFloat * halfBFloat;
    const float ac = aFloat * cFloat;
    const float discriminantFloat = halfBSquared - ac;
    if (finiteFloat(aFloat) && aFloat > 0.0f && finiteFloat(halfBFloat) &&
        finiteFloat(cFloat) && finiteFloat(halfBSquared) && finiteFloat(ac) &&
        finiteFloat(discriminantFloat))
    {
        const float cancellationBound =
            32.0f * FLT_EPSILON * (absolute(halfBSquared) + absolute(ac));
        if (discriminantFloat < 0.0f && -discriminantFloat > cancellationBound)
        {
            return false;
        }
        if (discriminantFloat > cancellationBound)
        {
            const float root = sqrtf(discriminantFloat);
            const float q = -halfBFloat - (halfBFloat < 0.0f ? -root : root);
            float nearT = q == 0.0f ? -halfBFloat / aFloat : q / aFloat;
            float farT = q == 0.0f ? nearT : cFloat / q;
            if (nearT > farT)
            {
                const float temporary = nearT;
                nearT = farT;
                farT = temporary;
            }
            t = nearT;
            if (!finiteFloat(t) || t < tMin || t > tMax)
            {
                t = farT;
            }
            if (finiteFloat(t) && t >= tMin && t <= tMax)
            {
                localPosition = localOrigin + t * localDirection;
                useDouble = !finiteVec3(localPosition) ||
                            !(maxAbsComponent(localPosition) > 0.0f);
            }
            else
            {
                return false;
            }
        }
    }

    if (useDouble)
    {
        // Far origins can erase the radius in float discriminant arithmetic.
        // The exceptional path keeps robust roots without taxing normal rays.
        const double a = dotDouble(localDirection, localDirection);
        const double halfB = dotDouble(localOrigin, localDirection);
        const double c = dotDouble(localOrigin, localOrigin) - 0.25;
        const double discriminant = halfB * halfB - a * c;
        if (!finiteDouble(a) || !(a > 0.0) || !finiteDouble(discriminant) ||
            discriminant < 0.0)
        {
            return false;
        }

        const double root = sqrt(discriminant);
        const double q = -halfB - (halfB < 0.0 ? -root : root);
        double nearT = q == 0.0 ? -halfB / a : q / a;
        double farT = q == 0.0 ? nearT : c / q;
        if (nearT > farT)
        {
            const double temporary = nearT;
            nearT = farT;
            farT = temporary;
        }

        if (!finiteDouble(nearT) || nearT < -static_cast<double>(FLT_MAX) ||
            nearT > static_cast<double>(FLT_MAX))
        {
            nearT = farT;
        }
        t = static_cast<float>(nearT);
        double selectedT = nearT;
        if (!finiteFloat(t) || t < tMin || t > tMax)
        {
            if (!finiteDouble(farT) || farT < -static_cast<double>(FLT_MAX) ||
                farT > static_cast<double>(FLT_MAX))
            {
                return false;
            }
            t = static_cast<float>(farT);
            selectedT = farT;
            if (!finiteFloat(t) || t < tMin || t > tMax)
            {
                return false;
            }
        }

        localPosition = glm::vec3(
            static_cast<float>(static_cast<double>(localOrigin.x) +
                               selectedT * static_cast<double>(localDirection.x)),
            static_cast<float>(static_cast<double>(localOrigin.y) +
                               selectedT * static_cast<double>(localDirection.y)),
            static_cast<float>(static_cast<double>(localOrigin.z) +
                               selectedT * static_cast<double>(localDirection.z)));
    }
    glm::vec3 localNormal;
    if (!safeNormalize(localPosition, localNormal))
    {
        return false;
    }

    glm::vec3 geometricNormal;
    if (!safeNormalize(transformVector(geom.invTranspose, localNormal), geometricNormal))
    {
        return false;
    }

    const float pi = 3.14159265358979323846f;
    const float theta = atan2f(localNormal.z, localNormal.x);
    const float phi = asinf(clampSignedUnit(localNormal.y));
    const glm::vec2 uv(theta / (2.0f * pi) + 0.5f, phi / pi + 0.5f);

    const float cosineTheta = cosf(theta);
    const float sineTheta = sinf(theta);
    const float cosinePhi = cosf(phi);
    const float sinePhi = sinf(phi);
    const glm::vec3 localDpdu = pi * cosinePhi *
                                    glm::vec3(-sineTheta, 0.0f, cosineTheta);
    const glm::vec3 localDpdv = 0.5f * pi *
                                    glm::vec3(-sinePhi * cosineTheta,
                                              cosinePhi,
                                              -sinePhi * sineTheta);
    glm::vec3 dpdu = transformVector(geom.transform, localDpdu);
    glm::vec3 dpdv = transformVector(geom.transform, localDpdv);
    validateTangents(geometricNormal, dpdu, dpdv);

    const glm::vec3 position = ray.origin + t * ray.direction;
    if (!finiteVec3(position))
    {
        return false;
    }
    finishHit(geom, primitiveId, ray, t, position, geometricNormal,
              geometricNormal, uv, dpdu, dpdv, hit);
    return true;
}

GEOMETRY_HD GEOMETRY_INLINE bool intersectCube(const Geom& geom,
                                                int primitiveId,
                                                const Ray& ray,
                                                float tMin,
                                                float tMax,
                                                Hit& hit)
{
    const glm::vec3 localOrigin = transformPoint(geom.inverseTransform, ray.origin);
    const glm::vec3 localDirection = transformVector(geom.inverseTransform, ray.direction);
    if (!finiteVec3(localOrigin) || !finiteVec3(localDirection) ||
        !(maxAbsComponent(localDirection) > 0.0f))
    {
        return false;
    }

    float enter = -FLT_MAX;
    float exit = FLT_MAX;
    glm::vec3 enterNormal(0.0f);
    glm::vec3 exitNormal(0.0f);

    for (int axis = 0; axis < 3; ++axis)
    {
        const float origin = localOrigin[axis];
        const float direction = localDirection[axis];
        if (direction == 0.0f)
        {
            if (origin < -0.5f || origin > 0.5f)
            {
                return false;
            }
            continue;
        }

        float axisEnter = (-0.5f - origin) / direction;
        float axisExit = (0.5f - origin) / direction;
        glm::vec3 axisEnterNormal(0.0f);
        glm::vec3 axisExitNormal(0.0f);
        axisEnterNormal[axis] = -1.0f;
        axisExitNormal[axis] = 1.0f;
        if (axisEnter > axisExit)
        {
            const float temporaryT = axisEnter;
            axisEnter = axisExit;
            axisExit = temporaryT;
            const glm::vec3 temporaryNormal = axisEnterNormal;
            axisEnterNormal = axisExitNormal;
            axisExitNormal = temporaryNormal;
        }

        // A nearly parallel direction can produce an unbounded slab interval.
        // Such an interval is harmless; NaNs cannot participate in clipping.
        if (axisEnter != axisEnter || axisExit != axisExit)
        {
            return false;
        }
        if (axisEnter > enter)
        {
            enter = axisEnter;
            enterNormal = axisEnterNormal;
        }
        if (axisExit < exit)
        {
            exit = axisExit;
            exitNormal = axisExitNormal;
        }
        if (enter > exit)
        {
            return false;
        }
    }

    float t = enter;
    glm::vec3 localNormal = enterNormal;
    if (!finiteFloat(t) || t < tMin || t > tMax)
    {
        t = exit;
        localNormal = exitNormal;
        if (!finiteFloat(t) || t < tMin || t > tMax)
        {
            return false;
        }
    }

    glm::vec3 geometricNormal;
    if (!safeNormalize(transformVector(geom.invTranspose, localNormal), geometricNormal))
    {
        return false;
    }

    const glm::vec3 localPosition = localOrigin + t * localDirection;
    glm::vec2 uv(0.0f);
    glm::vec3 localDpdu(0.0f);
    glm::vec3 localDpdv(0.0f);
    if (localNormal.x > 0.0f)
    {
        uv = glm::vec2(0.5f - localPosition.z, localPosition.y + 0.5f);
        localDpdu = glm::vec3(0.0f, 0.0f, -1.0f);
        localDpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (localNormal.x < 0.0f)
    {
        uv = glm::vec2(localPosition.z + 0.5f, localPosition.y + 0.5f);
        localDpdu = glm::vec3(0.0f, 0.0f, 1.0f);
        localDpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (localNormal.y > 0.0f)
    {
        uv = glm::vec2(localPosition.x + 0.5f, 0.5f - localPosition.z);
        localDpdu = glm::vec3(1.0f, 0.0f, 0.0f);
        localDpdv = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    else if (localNormal.y < 0.0f)
    {
        uv = glm::vec2(localPosition.x + 0.5f, localPosition.z + 0.5f);
        localDpdu = glm::vec3(1.0f, 0.0f, 0.0f);
        localDpdv = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    else if (localNormal.z > 0.0f)
    {
        uv = glm::vec2(localPosition.x + 0.5f, localPosition.y + 0.5f);
        localDpdu = glm::vec3(1.0f, 0.0f, 0.0f);
        localDpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        uv = glm::vec2(0.5f - localPosition.x, localPosition.y + 0.5f);
        localDpdu = glm::vec3(-1.0f, 0.0f, 0.0f);
        localDpdv = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    uv.x = clamp01(uv.x);
    uv.y = clamp01(uv.y);

    glm::vec3 dpdu = transformVector(geom.transform, localDpdu);
    glm::vec3 dpdv = transformVector(geom.transform, localDpdv);
    validateTangents(geometricNormal, dpdu, dpdv);
    const glm::vec3 position = ray.origin + t * ray.direction;
    if (!finiteVec3(position))
    {
        return false;
    }
    finishHit(geom, primitiveId, ray, t, position, geometricNormal,
              geometricNormal, uv, dpdu, dpdv, hit);
    return true;
}

static GEOMETRY_HD GEOMETRY_NOINLINE bool intersectPrimitive(const Geom& geom,
                                                              int primitiveId,
                                                              const Ray& ray,
                                                              float tMin,
                                                              float tMax,
                                                              Hit& hit)
{
    if (geom.type == TRIANGLE)
    {
        return intersectTriangle(geom, primitiveId, ray, tMin, tMax, hit);
    }
    if (geom.type == SPHERE)
    {
        return intersectSphere(geom, primitiveId, ray, tMin, tMax, hit);
    }
    if (geom.type == CUBE)
    {
        return intersectCube(geom, primitiveId, ray, tMin, tMax, hit);
    }
    return false;
}

GEOMETRY_HD GEOMETRY_INLINE bool isBetterHit(const Hit& candidate,
                                              float bestT,
                                              int bestPrimitive)
{
    return candidate.t < bestT ||
           (candidate.t == bestT && candidate.primitiveId < bestPrimitive);
}

GEOMETRY_HD GEOMETRY_INLINE float widenedSearchMaximum(float bestT, float requestedMaximum)
{
    if (!(bestT < requestedMaximum) || !finiteFloat(bestT))
    {
        return requestedMaximum;
    }
    // Search eight representable t values beyond the current winner to absorb
    // slab/root intermediate variation. Final ordering still uses exact float t
    // and primitive id, so this wider search cannot change the comparison key.
    float widened = bestT;
    for (int step = 0; step < 8 && widened < requestedMaximum; ++step)
    {
        widened = nextafterf(widened, FLT_MAX);
    }
    return widened < requestedMaximum ? widened : requestedMaximum;
}

GEOMETRY_HD GEOMETRY_INLINE bool closestBrute(const SceneView& scene,
                                               const Ray& ray,
                                               float tMin,
                                               float tMax,
                                               Hit& output)
{
    bool found = false;
    float bestT = tMax;
    int bestPrimitive = INT_MAX;
    Hit best;
    for (int primitive = 0; primitive < scene.geomCount; ++primitive)
    {
        Hit candidate;
        const float searchMaximum = widenedSearchMaximum(bestT, tMax);
        if (intersectPrimitive(scene.geoms[primitive], primitive, ray,
                               tMin, searchMaximum, candidate) &&
            isBetterHit(candidate, bestT, bestPrimitive))
        {
            found = true;
            bestT = candidate.t;
            bestPrimitive = primitive;
            best = candidate;
        }
    }
    if (found)
    {
        output = best;
    }
    return found;
}

GEOMETRY_HD GEOMETRY_INLINE bool anyBrute(const SceneView& scene,
                                           const Ray& ray,
                                           float tMin,
                                           float tMax)
{
    for (int primitive = 0; primitive < scene.geomCount; ++primitive)
    {
        Hit candidate;
        if (intersectPrimitive(scene.geoms[primitive], primitive, ray,
                               tMin, tMax, candidate))
        {
            return true;
        }
    }
    return false;
}

GEOMETRY_HD GEOMETRY_INLINE bool validBounds(const BvhNode& node)
{
    return finiteVec3(node.boundsMin) && finiteVec3(node.boundsMax) &&
           node.boundsMin.x <= node.boundsMax.x &&
           node.boundsMin.y <= node.boundsMax.y &&
           node.boundsMin.z <= node.boundsMax.z;
}

GEOMETRY_HD GEOMETRY_INLINE bool validNodeShape(const SceneView& scene,
                                                 int nodeIndex,
                                                 int depth)
{
    const BvhNode& node = scene.nodes[nodeIndex];
    if (node.count > 0)
    {
        if (node.left != -1 || node.right != -1 || node.first < 0 ||
            node.count > scene.geomCount || node.first > scene.geomCount - node.count)
        {
            return false;
        }
        // Check indices even when this leaf's bounds miss the ray. Otherwise a
        // malformed leaf can be mistaken for a valid cull and silently hide a
        // primitive that brute force would hit.
        for (int offset = 0; offset < node.count; ++offset)
        {
            const int primitive = scene.primitiveIndices[node.first + offset];
            if (primitive < 0 || primitive >= scene.geomCount)
            {
                return false;
            }
        }
        return true;
    }

    if (node.count != 0 || depth >= 60 || node.left < 0 ||
        node.left >= scene.nodeCount || node.right < 0 ||
        node.right >= scene.nodeCount || node.left == node.right ||
        node.left == nodeIndex || node.right == nodeIndex)
    {
        return false;
    }
    return validBounds(scene.nodes[node.left]) && validBounds(scene.nodes[node.right]);
}

GEOMETRY_HD GEOMETRY_INLINE bool intersectBounds(const BvhNode& node,
                                                  const Ray& ray,
                                                  float tMin,
                                                  float tMax,
                                                  float& nearValue,
                                                  float& farValue)
{
    float nearT = tMin;
    float farT = tMax;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float direction = ray.direction[axis];
        const float origin = ray.origin[axis];
        if (direction == 0.0f)
        {
            if (origin < node.boundsMin[axis] || origin > node.boundsMax[axis])
            {
                return false;
            }
            continue;
        }

        float axisNear = (node.boundsMin[axis] - origin) / direction;
        float axisFar = (node.boundsMax[axis] - origin) / direction;
        if (axisNear > axisFar)
        {
            const float temporary = axisNear;
            axisNear = axisFar;
            axisFar = temporary;
        }
        if (axisNear != axisNear || axisFar != axisFar)
        {
            return false;
        }
        if (axisNear > nearT)
        {
            nearT = axisNear;
        }
        if (axisFar < farT)
        {
            farT = axisFar;
        }
        if (nearT > farT)
        {
            return false;
        }
    }
    nearValue = nearT;
    farValue = farT;
    return true;
}

struct StackEntry
{
    int node;
    int depth;
};

GEOMETRY_HD GEOMETRY_INLINE bool closestBvh(const SceneView& scene,
                                             const Ray& ray,
                                             float tMin,
                                             float tMax,
                                             Hit& output,
                                             bool& valid)
{
    valid = true;
    StackEntry stack[64];
    int stackSize = 1;
    stack[0].node = 0;
    stack[0].depth = 0;

    bool found = false;
    float bestT = tMax;
    int bestPrimitive = INT_MAX;
    Hit best;

    while (stackSize > 0)
    {
        const StackEntry entry = stack[--stackSize];
        if (entry.node < 0 || entry.node >= scene.nodeCount || entry.depth > 60)
        {
            valid = false;
            return false;
        }

        const BvhNode& node = scene.nodes[entry.node];
        if (!validBounds(node) || !validNodeShape(scene, entry.node, entry.depth))
        {
            valid = false;
            return false;
        }
        float nodeNear = 0.0f;
        float nodeFar = 0.0f;
        const float nodeSearchMaximum = widenedSearchMaximum(bestT, tMax);
        if (!intersectBounds(node, ray, tMin, nodeSearchMaximum, nodeNear, nodeFar))
        {
            continue;
        }

        if (node.count > 0)
        {
            for (int offset = 0; offset < node.count; ++offset)
            {
                const int primitive = scene.primitiveIndices[node.first + offset];
                Hit candidate;
                const float primitiveSearchMaximum = widenedSearchMaximum(bestT, tMax);
                if (intersectPrimitive(scene.geoms[primitive], primitive, ray,
                                       tMin, primitiveSearchMaximum, candidate) &&
                    isBetterHit(candidate, bestT, bestPrimitive))
                {
                    found = true;
                    bestT = candidate.t;
                    bestPrimitive = primitive;
                    best = candidate;
                }
            }
            continue;
        }

        const BvhNode& left = scene.nodes[node.left];
        const BvhNode& right = scene.nodes[node.right];

        float leftNear = 0.0f;
        float leftFar = 0.0f;
        float rightNear = 0.0f;
        float rightFar = 0.0f;
        const float childSearchMaximum = widenedSearchMaximum(bestT, tMax);
        const bool hitLeft = intersectBounds(left, ray, tMin, childSearchMaximum,
                                             leftNear, leftFar);
        const bool hitRight = intersectBounds(right, ray, tMin, childSearchMaximum,
                                              rightNear, rightFar);
        const int additions = (hitLeft ? 1 : 0) + (hitRight ? 1 : 0);
        if (stackSize + additions > 64)
        {
            valid = false;
            return false;
        }

        if (hitLeft && hitRight)
        {
            const bool leftFirst = leftNear < rightNear ||
                                   (leftNear == rightNear && node.left < node.right);
            const StackEntry nearEntry = {leftFirst ? node.left : node.right, entry.depth + 1};
            const StackEntry farEntry = {leftFirst ? node.right : node.left, entry.depth + 1};
            stack[stackSize++] = farEntry;
            stack[stackSize++] = nearEntry;
        }
        else if (hitLeft)
        {
            stack[stackSize++] = {node.left, entry.depth + 1};
        }
        else if (hitRight)
        {
            stack[stackSize++] = {node.right, entry.depth + 1};
        }
    }

    if (found)
    {
        output = best;
    }
    return found;
}

GEOMETRY_HD GEOMETRY_INLINE bool anyBvh(const SceneView& scene,
                                         const Ray& ray,
                                         float tMin,
                                         float tMax,
                                         bool& valid)
{
    valid = true;
    StackEntry stack[64];
    int stackSize = 1;
    stack[0] = {0, 0};

    while (stackSize > 0)
    {
        const StackEntry entry = stack[--stackSize];
        if (entry.node < 0 || entry.node >= scene.nodeCount || entry.depth > 60)
        {
            valid = false;
            return false;
        }
        const BvhNode& node = scene.nodes[entry.node];
        if (!validBounds(node) || !validNodeShape(scene, entry.node, entry.depth))
        {
            valid = false;
            return false;
        }
        float nodeNear = 0.0f;
        float nodeFar = 0.0f;
        if (!intersectBounds(node, ray, tMin, tMax, nodeNear, nodeFar))
        {
            continue;
        }

        if (node.count > 0)
        {
            for (int offset = 0; offset < node.count; ++offset)
            {
                const int primitive = scene.primitiveIndices[node.first + offset];
                Hit candidate;
                if (intersectPrimitive(scene.geoms[primitive], primitive, ray,
                                       tMin, tMax, candidate))
                {
                    return true;
                }
            }
            continue;
        }

        const BvhNode& left = scene.nodes[node.left];
        const BvhNode& right = scene.nodes[node.right];
        float leftNear = 0.0f;
        float leftFar = 0.0f;
        float rightNear = 0.0f;
        float rightFar = 0.0f;
        const bool hitLeft = intersectBounds(left, ray, tMin, tMax, leftNear, leftFar);
        const bool hitRight = intersectBounds(right, ray, tMin, tMax, rightNear, rightFar);
        const int additions = (hitLeft ? 1 : 0) + (hitRight ? 1 : 0);
        if (stackSize + additions > 64)
        {
            valid = false;
            return false;
        }
        if (hitLeft && hitRight)
        {
            const bool leftFirst = leftNear < rightNear ||
                                   (leftNear == rightNear && node.left < node.right);
            stack[stackSize++] = {leftFirst ? node.right : node.left, entry.depth + 1};
            stack[stackSize++] = {leftFirst ? node.left : node.right, entry.depth + 1};
        }
        else if (hitLeft)
        {
            stack[stackSize++] = {node.left, entry.depth + 1};
        }
        else if (hitRight)
        {
            stack[stackSize++] = {node.right, entry.depth + 1};
        }
    }
    return false;
}

} // namespace geometry_detail

GEOMETRY_HD GEOMETRY_INLINE bool closestHit(const SceneView& scene,
                                             const Ray& ray,
                                             float tMin,
                                             float tMax,
                                             bool useBvh,
                                             Hit& hit)
{
    if (!(tMin <= tMax) || scene.geomCount < 0 ||
        (scene.geomCount > 0 && scene.geoms == nullptr) ||
        !geometry_detail::validRay(ray))
    {
        return false;
    }
    if (scene.geomCount == 0)
    {
        return false;
    }

    if (!useBvh || scene.nodes == nullptr || scene.nodeCount <= 0 ||
        scene.primitiveIndices == nullptr)
    {
        return geometry_detail::closestBrute(scene, ray, tMin, tMax, hit);
    }

    bool valid = false;
    const bool found = geometry_detail::closestBvh(scene, ray, tMin, tMax, hit, valid);
    if (valid)
    {
        return found;
    }
    return geometry_detail::closestBrute(scene, ray, tMin, tMax, hit);
}

GEOMETRY_HD GEOMETRY_INLINE bool anyHit(const SceneView& scene,
                                         const Ray& ray,
                                         float tMin,
                                         float tMax,
                                         bool useBvh)
{
    if (!(tMin <= tMax) || scene.geomCount < 0 ||
        (scene.geomCount > 0 && scene.geoms == nullptr) ||
        !geometry_detail::validRay(ray))
    {
        return false;
    }
    if (scene.geomCount == 0)
    {
        return false;
    }
    if (!useBvh || scene.nodes == nullptr || scene.nodeCount <= 0 ||
        scene.primitiveIndices == nullptr)
    {
        return geometry_detail::anyBrute(scene, ray, tMin, tMax);
    }

    bool valid = false;
    const bool found = geometry_detail::anyBvh(scene, ray, tMin, tMax, valid);
    if (found)
    {
        return true;
    }
    return valid ? false : geometry_detail::anyBrute(scene, ray, tMin, tMax);
}

#undef GEOMETRY_INLINE
#undef GEOMETRY_NOINLINE
#undef GEOMETRY_HD
