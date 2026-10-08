#pragma once

#include <glm/glm.hpp>
#include "scene.h"

struct Photon
{
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 power;
    bool passedGlass;
    bool dispersed; 
    int remainingBounces;
    int channel;      // 0: R, 1: G, 2: B
};


void photonMap(
    Scene* scene, int numPhotons, int iter,
    Geom* geoms, int geomsSize,
    Material* materials, int materialsSize,
    Camera cam, glm::vec3* image, const float* camDepth);
