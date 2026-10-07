#pragma once

#include "camera.h"
#include "render_settings.h"
#include "scene_structs.h"
#include <string>
#include <vector>

struct RenderState {
    CameraData camera;
    CameraConfig controls;
    unsigned int iterations;
    std::vector<glm::vec3> image;
    std::string image_name;
    RenderSettings settings;
};

struct EnvironmentMap {
    int width = 0;
    int height = 0;
    std::string path;
    std::vector<glm::vec3> pixels;
};

inline constexpr const char* HDRI_DIR = "assets/hdri";

class Scene {
  private:
    void load_hdri_pixels(const std::string& json_name, const std::string& hdri_path);
    void load_from_json(const std::string& json_name);
    void scan_hdris();

  public:
    Scene(std::string filename);
    void save_to_json(const std::string& out_path) const;
    void set_environment(const std::string& hdri_path);

    std::string filename;

    std::vector<Geom> geoms;
    std::vector<std::string> material_names;
    std::vector<Material> materials;
    std::vector<Light> lights;
    std::vector<std::string> hdri_names;

    RenderState state;
    EnvironmentMap env;
};
