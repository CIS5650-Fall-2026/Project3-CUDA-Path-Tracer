#pragma once

#include "sceneStructs.h"
#include "utilities.h"

__host__ __device__ inline Ray sampleLens(const Camera& camera, Ray ray, float u, float v) {
    if (camera.apertureRadius == 0.0f) {
        return ray;
    }
    glm::vec3 forward = glm::normalize(camera.view);
    glm::vec3 right = glm::normalize(glm::cross(forward, camera.up));
    glm::vec3 up = glm::normalize(glm::cross(right, forward));
    glm::vec3 focus = ray.origin + ray.direction
        * (camera.focalDistance / glm::dot(ray.direction, forward));

    // sqrt gives uniform area samples rather than clustering at the lens center.
    float radius = camera.apertureRadius * sqrtf(u);
    float angle = TWO_PI * v;
    ray.origin += radius * (cosf(angle) * right + sinf(angle) * up);
    ray.direction = glm::normalize(focus - ray.origin);
    return ray;
}
