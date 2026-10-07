#pragma once

struct RenderSettings {
    int max_depth;
    int sdf_max_steps;
    float sdf_hit_eps;
    float sdf_normal_eps;
    float env_strength;
    bool agx;
};
