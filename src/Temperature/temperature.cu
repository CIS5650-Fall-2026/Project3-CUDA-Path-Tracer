
#include "temperature.h"
#include <glm/glm.hpp>

// Modified version of https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
// To support temperature from 0 to 40000 kelvin
 __device__ glm::vec3 kelvin_to_rgb(float temperature) {
     float t = fmin(40000.0f, temperature) / 100.0f;

     float r, g, b;
     const float inv = 1.0f / 255.0f;
    
     if(t <= 66.0f) {
         r = 1.0f;
     } else {
         r = __saturatef(inv * 329.698727446 * powf(t - 60.0f, -0.1332047592));
     }

     if (t <= 66.0f) {
         g = __saturatef(inv * (99.4708025861 * logf(t) - 161.1195681661));
     }
     else {
         g = __saturatef(inv * 288.1221695283 * powf(t - 60.0f, -0.0755148492));
     }

     if (t < 10.0f) {
         // Jank way to linearly go to 0
         g = fmaxf(g, 0.0f);
         return glm::smoothstep(3.0f, 10.0f, t) * glm::vec3(r, g, 0.0f);
     }

     if (t >= 66.0f) {
         b = 1.0f;
     }
     else if(t <= 19.0f) {
         b = 0.0f;
     }
     else {
         b = __saturatef(inv * (138.5177312231 * logf(t - 10.0f) - 305.0447927307));
     }

     return glm::vec3(r, g, b);
 }
