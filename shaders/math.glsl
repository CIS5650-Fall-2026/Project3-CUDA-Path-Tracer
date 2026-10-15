#ifndef MATH_GLSL
#define MATH_GLSL

// Numeric constants
const float kPi = 3.14159265358979323;
const float kInvPi = 0.31830988618379067;
const float kHalfPi = 1.57079632679489662;
const float kOneMinusEpsilon = 0.99999994;
const float kFloatEqualityEpsilon = 0.000005;

// Data structures
struct Ray {
    vec3 origin;
    vec3 direction;
};

float absDot(vec3 a, vec3 b)
{
    return abs(dot(a, b));
}

Ray spawnRay(vec3 pos, vec3 wi)
{
    return Ray(pos + wi * 0.0001, wi);
}

bool refractDir(vec3 wi, vec3 n, float eta, out vec3 wt)
{
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1)
    {
        return false;
    }

    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

vec3 faceForward(vec3 n, vec3 v)
{
    return (dot(n, v) < 0.f) ? -n : n;
}

void coordinateSystem(vec3 v1, out vec3 v2, out vec3 v3)
{
    if (abs(v1.x) > abs(v1.y))
        v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

mat3 localToWorld(vec3 nor)
{
    vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return mat3(tan, bit, nor);
}

mat3 worldToLocal(vec3 nor)
{
    return transpose(localToWorld(nor));
}

bool areEqual(float x, float y)
{
    return abs(x - y) < kFloatEqualityEpsilon;
}

#endif