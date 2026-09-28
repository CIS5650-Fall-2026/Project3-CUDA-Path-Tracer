#pragma once

#include "intersections.h"
#include "utilities.h"

struct LightSample {
    glm::vec3 position;
    glm::vec3 normal;
    float areaPdf;
};

__host__ __device__ inline float powerWeight(float pdf, float otherPdf) {
    if (pdf <= 0.0f) return 0.0f;
    if (otherPdf <= 0.0f) return 1.0f;
    // Use a ratio to avoid squaring a very large directional PDF.
    if (pdf >= otherPdf) {
        float ratio = otherPdf / pdf;
        return 1.0f / (1.0f + ratio * ratio);
    }
    float ratio = pdf / otherPdf;
    return ratio * ratio / (1.0f + ratio * ratio);
}

__host__ __device__ inline float lightAreaPdf(const Geom& light, const Triangle* triangles,
    glm::vec3 point, int triangleIndex) {
    if (light.type == MESH) {
        if (triangleIndex < light.triangleStart || triangleIndex >= light.triangleStart + light.triangleCount)
            return 0.0f;
        const Triangle& triangle = triangles[triangleIndex];
        float area = 0.5f * glm::length(glm::cross(triangle.b - triangle.a, triangle.c - triangle.a));
        return area > 0 ? 1.0f / (light.triangleCount * area) : 0.0f;
    }
    glm::vec3 localPoint = glm::vec3(light.inverseTransform * glm::vec4(point, 1.0f));
    if (light.type == CUBE) {
        glm::vec3 magnitude = glm::abs(localPoint);
        int axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0
            : magnitude.y >= magnitude.z ? 1 : 2;
        float area = glm::length(glm::cross(glm::vec3(light.transform[(axis + 1) % 3]),
            glm::vec3(light.transform[(axis + 2) % 3])));
        return area > 0 ? 1.0f / (6.0f * area) : 0.0f;
    }
    glm::vec3 normal = glm::normalize(localPoint);
    float areaScale = fabsf(glm::determinant(glm::mat3(light.transform)))
        * glm::length(glm::vec3(light.invTranspose * glm::vec4(normal, 0.0f)));
    return areaScale > 0 ? 1.0f / (PI * areaScale) : 0.0f;
}

// The PDF is measured per unit of world-space surface area.
__host__ __device__ inline LightSample sampleLightSurface(const Geom& light,
    const Triangle* triangles, float choice, float u, float v) {
    LightSample sample{};
    if (light.type == MESH) {
        if (light.triangleCount == 0) return sample;
        int index = glm::min(int(choice * light.triangleCount), light.triangleCount - 1);
        const Triangle& triangle = triangles[light.triangleStart + index];
        float root = sqrtf(u);
        sample.position = (1.0f - root) * triangle.a + root * (1.0f - v) * triangle.b
            + root * v * triangle.c;
        sample.normal = triangle.normal;
        float area = 0.5f * glm::length(glm::cross(triangle.b - triangle.a, triangle.c - triangle.a));
        if (area > 0.0f) sample.areaPdf = 1.0f / (light.triangleCount * area);
        return sample;
    }

    glm::vec3 point(0.0f), normal(0.0f);
    if (light.type == CUBE) {
        int face = glm::min(int(choice * 6), 5);
        int axis = face / 2;
        int a = (axis + 1) % 3;
        int b = (axis + 2) % 3;
        normal[axis] = face % 2 == 0 ? -1.0f : 1.0f;
        point[axis] = normal[axis] * 0.5f;
        point[a] = u - 0.5f;
        point[b] = v - 0.5f;
        float area = glm::length(glm::cross(glm::vec3(light.transform[a]), glm::vec3(light.transform[b])));
        if (area > 0.0f) sample.areaPdf = 1.0f / (6.0f * area);
    } else {
        float z = 1.0f - 2.0f * u;
        float radius = sqrtf(glm::max(0.0f, 1.0f - z * z));
        normal = glm::vec3(radius * cosf(TWO_PI * v), radius * sinf(TWO_PI * v), z);
        point = normal * 0.5f;
        // Nonuniform scale turns the sphere into an ellipsoid; account for the local area change.
        float areaScale = fabsf(glm::determinant(glm::mat3(light.transform)))
            * glm::length(glm::vec3(light.invTranspose * glm::vec4(normal, 0.0f)));
        if (areaScale > 0.0f) sample.areaPdf = 1.0f / (PI * areaScale);
    }
    sample.position = glm::vec3(light.transform * glm::vec4(point, 1.0f));
    sample.normal = glm::normalize(glm::vec3(light.invTranspose * glm::vec4(normal, 0.0f)));
    return sample;
}

__host__ __device__ inline bool shadowBlocked(Ray ray, float maxDistance,
    const Geom* geoms, int geomCount, const Triangle* triangles, bool meshCulling) {
    for (int i = 0; i < geomCount; i++) {
        const Geom& geom = geoms[i];
        glm::vec3 point, normal;
        bool outside;
        float distance;
        if (geom.type == CUBE) {
            distance = boxIntersectionTest(geom, ray, point, normal, outside);
        } else if (geom.type == SPHERE) {
            distance = sphereIntersectionTest(geom, ray, point, normal, outside);
        } else {
            distance = meshIntersectionTest(geom, triangles, ray, normal, meshCulling, maxDistance);
        }
        if (distance > 0.0f && distance < maxDistance) return true;
    }
    return false;
}
