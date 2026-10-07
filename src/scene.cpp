#include "scene.h"

#include "config.h"
#include "utilities.h"

#include "json.hpp"
#include <glm/gtc/matrix_inverse.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <charconv>
#include <filesystem>
#include <stb_image.h>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace nlohmann {
template <> struct adl_serializer<float> {
    static void to_json(json& j, float f) {
        char buf[32]{};
        std::to_chars(buf, buf + sizeof(buf) - 1, f);
        j = std::strtod(buf, nullptr);
    }
    static void from_json(const json& j, float& f) {
        f = j.get<double>();
    }
};

// glm vec serialiazer
template <glm::length_t L, typename T, glm::qualifier Q> struct adl_serializer<glm::vec<L, T, Q>> {
    static void to_json(json& j, const glm::vec<L, T, Q>& v) {
        j = json::array();
        for (glm::length_t i = 0; i < L; ++i) {
            j.push_back(v[i]);
        }
    }
    static void from_json(const json& j, glm::vec<L, T, Q>& v) {
        for (glm::length_t i = 0; i < L; ++i) {
            v[i] = j.at(i).get<T>();
        }
    }
};
} // namespace nlohmann

// struct macros
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Emission, color, strength)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Lambertian, color)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Conductor, eta, k, roughness, anisotropy)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Dielectric, ior, roughness)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DisneyDiffuse, color, roughness, subsurface)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DisneyMetal, color, edge_tint, roughness,
                                                anisotropic)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DisneyClearcoat, gloss)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DisneyGlass, color, roughness, anisotropic, ior)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DisneySheen, color, sheen_tint)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DisneyBsdf, color, specular_transmission, metallic,
                                                edge_tint, subsurface, specular, roughness,
                                                specular_tint, anisotropic, sheen, sheen_tint,
                                                clearcoat, clearcoat_gloss, ior)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RenderSettings, max_depth, sdf_max_steps, sdf_hit_eps,
                                   sdf_normal_eps, agx)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CameraConfig, accel, damping, mouse_sens,
                                                scroll_sens)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Transform, translation, rotation, scale)
NLOHMANN_JSON_SERIALIZE_ENUM(GeomType,
                             {{Sphere, "sphere"}, {Cube, "cube"}, {Plane, "plane"}, {Sdf, "sdf"}})

static constexpr const char* bsdf_types[] = {"diffuse",        "conductor",    "dielectric",
                                             "disney diffuse", "disney metal", "disney clearcoat",
                                             "disney glass",   "disney sheen", "disney bsdf"};

void to_json(json& j, const Material& m) {
    cuda::std::visit([&](const auto& b) { j = b; }, m.bsdf);
    j["type"] = bsdf_types[m.bsdf.index()];
    j["emission"] = m.emission;
}

void from_json(const json& j, Material& m) {
    const std::string type = j.at("type");
    if (type == "diffuse") {
        m.bsdf = j.get<Lambertian>();
    } else if (type == "conductor") {
        m.bsdf = j.get<Conductor>();
    } else if (type == "dielectric") {
        m.bsdf = j.get<Dielectric>();
    } else if (type == "disney diffuse") {
        m.bsdf = j.get<DisneyDiffuse>();
    } else if (type == "disney metal") {
        m.bsdf = j.get<DisneyMetal>();
    } else if (type == "disney clearcoat") {
        m.bsdf = j.get<DisneyClearcoat>();
    } else if (type == "disney glass") {
        m.bsdf = j.get<DisneyGlass>();
    } else if (type == "disney sheen") {
        m.bsdf = j.get<DisneySheen>();
    } else if (type == "disney bsdf") {
        m.bsdf = j.get<DisneyBsdf>();
    } else {
        throw std::runtime_error("unknown material type " + type);
    }
    m.emission = j.value("emission", Emission{});
}

void to_json(json& j, const CameraData& c) {
    j = {{"resolution", c.resolution},
         {"position", c.position},
         {"look_at", c.position + c.view},
         {"up", c.up},
         {"fovy", c.fov.y},
         {"lens_radius", c.lens_radius},
         {"focal_distance", c.focal_distance}};
}

void from_json(const json& j, CameraData& c) {
    j.at("resolution").get_to(c.resolution);
    j.at("position").get_to(c.position);
    j.at("look_at").get_to(c.look_at);
    j.at("up").get_to(c.up);
    j.at("lens_radius").get_to(c.lens_radius);
    j.at("focal_distance").get_to(c.focal_distance);
    c.set_fov(j.at("fovy").get<float>());
    c.view = glm::normalize(c.look_at - c.position);
    c.right = glm::normalize(glm::cross(c.view, c.up));
}

Scene::Scene(string filename) : filename(filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        load_from_json(filename);
        scan_hdris();
        return;
    } else {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::load_hdri_pixels(const std::string& json_name, const std::string& image_name) {
    const std::filesystem::path scene_path = std::filesystem::absolute(json_name);
    const std::filesystem::path image_path =
        (scene_path.parent_path() / image_name).lexically_normal();
    std::cout << "loading hdri: " << image_path.string() << '\n';

    int source_channels{};

    const std::string filename = image_path.string();
    float* decoded = stbi_loadf(filename.c_str(), &env.width, &env.height, &source_channels, 3);
    if (decoded == nullptr) {
        const char* reason = stbi_failure_reason();

        throw std::runtime_error("failed to load hdir " + filename + "\n" +
                                 (reason ? reason : "unknown error"));
    }

    const size_t pixel_count = static_cast<size_t>(env.width) * static_cast<size_t>(env.height);
    env.pixels.resize(pixel_count);

    for (size_t i = 0; i < pixel_count; ++i) {
        env.pixels[i] = glm::vec3(decoded[3 * i + 0], decoded[3 * i + 1], decoded[3 * i + 2]);
    }

    stbi_image_free(decoded);
}

void Scene::load_from_json(const std::string& json_name) {
    std::ifstream f(json_name);
    json data = json::parse(f);

    data.at("file").get_to(state.image_name);
    data.at("iterations").get_to(state.iterations);
    state.settings = data.value("settings", default_render_settings);
    state.settings.env_strength = 0.f;
    if (data.contains("environment")) {
        const auto& e = data["environment"];
        e.at("path").get_to(env.path);
        state.settings.env_strength = e.at("strength");
        load_hdri_pixels(json_name, env.path);
    }

    std::unordered_map<std::string, int> mat_name_to_id;
    for (const auto& [name, m] : data.at("materials").items()) {
        mat_name_to_id[name] = materials.size();
        material_names.push_back(name);
        materials.push_back(m.get<Material>());
    }

    for (const auto& o : data.at("objects")) {
        Geom new_geom{};
        o.at("type").get_to(new_geom.type);
        if (o.contains("material") == o.contains("light")) {
            throw std::runtime_error("object needs exactly one of material or light");
        }
        if (o.contains("light")) {
            if (new_geom.type != Plane) {
                throw std::runtime_error("light geoms must be planes");
            }
            new_geom.role = GeomRole::Light;
            new_geom.id = lights.size();
            lights.push_back({o["light"].get<Emission>(), static_cast<int>(geoms.size())});
        } else {
            new_geom.role = GeomRole::Material;
            new_geom.id = mat_name_to_id.at(o["material"].get<std::string>());
        }
        Transform& t = new_geom.transform;
        o.at("transform").get_to(t);
        t.matrix = utility_core::build_transformation_matrix(t.translation, t.rotation, t.scale);
        t.inverse = glm::inverse(t.matrix);
        t.inv_transpose = glm::inverseTranspose(t.matrix);
        geoms.push_back(new_geom);
    }

    data.at("camera").get_to(state.camera);
    state.controls = data.value("controls", CameraConfig{});
    state.image.assign(state.camera.resolution.x * state.camera.resolution.y, glm::vec3(0));
}

void Scene::save_to_json(const std::string& out_path) const {
    const std::filesystem::path out_dir = std::filesystem::absolute(out_path).parent_path();
    json data;
    data["file"] = state.image_name;
    data["iterations"] = state.iterations;
    data["settings"] = state.settings;
    if (!env.path.empty()) {
        const auto src_dir = std::filesystem::absolute(filename).parent_path();
        data["environment"] = {
            {"path", std::filesystem::relative(src_dir / env.path, out_dir).generic_string()},
            {"strength", state.settings.env_strength}};
    }
    for (size_t i = 0; i < materials.size(); ++i) {
        data["materials"][material_names[i]] = materials[i];
    }
    for (const Geom& g : geoms) {
        json o = {{"type", g.type}, {"transform", g.transform}};
        if (g.role == GeomRole::Light) {
            o["light"] = lights[g.id].emission;
        } else {
            o["material"] = material_names[g.id];
        }
        data["objects"].push_back(o);
    }
    data["camera"] = state.camera;
    data["controls"] = state.controls;

    std::filesystem::create_directories(out_dir);
    std::ofstream(out_path) << data.dump(4);
}

void Scene::scan_hdris() {
    std::error_code ec;
    for (const auto& e : std::filesystem::recursive_directory_iterator(HDRI_DIR, ec)) {
        if (e.path().extension() == ".hdr") {
            hdri_names.push_back(e.path().lexically_relative(HDRI_DIR).generic_string());
        }
    }
}

void Scene::set_environment(const std::string& hdri_path) {
    env = {};
    if (hdri_path.empty()) {
        return;
    }
    const auto scene_dir = std::filesystem::absolute(filename).parent_path();
    const std::string path =
        std::filesystem::proximate(std::filesystem::absolute(hdri_path), scene_dir)
            .generic_string();
    load_hdri_pixels(filename, path);
    env.path = path;
}
