#include "scene.h"
#include "json.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    const auto root = std::filesystem::temp_directory_path() / ("pathtracer-scene-" +
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    try {
        require(argc == 2, "Expected cornell_glass.json path");
        std::ifstream source(argv[1]); require(bool(source), "Cannot read test scene");
        json base = json::parse(source);
        std::filesystem::create_directories(root);
        auto load = [&](const json& value) {
            const auto p = root / "test.json";
            { std::ofstream out(p); out << value.dump(2); }
            return Scene(p.string());
        };
        auto reject = [&](const json& value) {
            bool threw = false;
            try { load(value); } catch (const std::exception&) { threw = true; }
            require(threw, "Parser accepted invalid material/assignment");
        };
        const Scene initial = load(base);
        require(initial.state.compactPaths && initial.state.antialiasing, "Default toggles changed");
        int glass = -1, metal = -1;
        for (int i = 0; i < static_cast<int>(initial.materials.size()); ++i) {
            if (initial.materials[i].type == MaterialType::Transmission) glass = i;
            if (initial.materials[i].type == MaterialType::Metal) metal = i;
        }
        require(glass >= 0 && metal >= 0, "Fixture needs glass and metal");
        require(initial.materials[glass].transmission.ior == 1.5f, "Glass preset changed");
        require(initial.materials[metal].Kd == glm::vec3(0) && initial.materials[metal].Ks == glm::vec3(1),
            "Metal default coefficients changed");
        require(Material::roughnessToAlpha(0) == Material::roughnessToAlpha(0.001f),
            "Roughness clamp changed");
        auto changed = base;
        changed["compactPaths"] = false; changed["antialiasing"] = false;
        const auto disabled = load(changed);
        require(!disabled.state.compactPaths && !disabled.state.antialiasing, "Flags were not parsed");
        require(disabled.checkpointSceneIdentity != initial.checkpointSceneIdentity,
            "Sampling flags missing from checkpoint identity");
        changed = base; changed["sortMaterials"] = false;
        changed["Camera"]["FILE"] = "different"; changed["Camera"]["ITERATIONS"] = 321;
        require(load(changed).checkpointSceneIdentity == initial.checkpointSceneIdentity,
            "Output/target/sort unexpectedly changed identity");
        changed = base; changed["materials"][glass]["roughness"] = 0; reject(changed);
        changed = base; changed["materials"][glass]["transmissionPresetID"] = 1; reject(changed);
        changed = base; changed["materials"][metal]["Kd"] = 0; reject(changed);
        changed = base; changed["materials"][metal]["metalPresetID"] = -1; reject(changed);
        changed = base; changed["Objects"][0]["material"] = 999; reject(changed);
        const auto generated = std::filesystem::path(argv[1]).parent_path() / "generated";
        if (std::filesystem::exists(generated)) {
            for (const auto& entry : std::filesystem::directory_iterator(generated)) {
                if (entry.path().extension() == ".json") {
                    const Scene parsed(entry.path().string());
                    require(!parsed.geoms.empty() && !parsed.materials.empty(), "Empty generated scene");
                }
            }
        }
        std::filesystem::remove_all(root);
        std::cout << "PASS: production scene parser, presets, flags and checkpoint identity (CPU)\n";
        return 0;
    } catch (const std::exception& e) {
        std::error_code ignored; std::filesystem::remove_all(root, ignored);
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
