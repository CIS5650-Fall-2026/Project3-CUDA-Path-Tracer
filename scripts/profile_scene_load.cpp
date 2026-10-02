// Original measurement helper for the project's own scene loader and BVH.
// Compile against cis565_core; this does not implement a second renderer.
#include "scene.h"
#include <json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2)
            throw std::runtime_error("Supply one or more scene JSON paths.");
        using Clock = std::chrono::steady_clock;
        nlohmann::json output = nlohmann::json::array();
        for (int argument = 1; argument < argc; ++argument)
        {
            std::vector<double> loadTimes, rebuildTimes;
            std::size_t primitives = 0, triangles = 0, nodeCount = 0;
            // One warmup followed by three measurements of both operations.
            for (int repeat = -1; repeat < 3; ++repeat)
            {
                const auto start = Clock::now();
                Scene scene(argv[argument]);
                const auto loaded = Clock::now();
                std::vector<BvhNode> nodes;
                std::vector<int> indices;
                buildBvh(scene.geoms, nodes, indices, scene.bvhMaxDepth);
                const auto rebuilt = Clock::now();
                primitives = scene.geoms.size();
                triangles = static_cast<std::size_t>(std::count_if(
                    scene.geoms.begin(), scene.geoms.end(),
                    [](const Geom& geom) { return geom.type == TRIANGLE; }));
                nodeCount = scene.bvhNodes.size();
                if (repeat >= 0)
                {
                    loadTimes.push_back(std::chrono::duration<double, std::milli>(loaded - start).count());
                    rebuildTimes.push_back(std::chrono::duration<double, std::milli>(rebuilt - loaded).count());
                }
            }
            auto median = [](std::vector<double> values)
            {
                std::sort(values.begin(), values.end());
                return values[values.size() / 2];
            };
            output.push_back({{"scene", argv[argument]}, {"primitives", primitives},
                {"triangles", triangles}, {"bvh_nodes", nodeCount},
                {"load_including_bvh_ms", loadTimes}, {"bvh_rebuild_ms", rebuildTimes},
                {"load_including_bvh_median_ms", median(loadTimes)},
                {"bvh_rebuild_median_ms", median(rebuildTimes)}});
        }
        std::cout << output.dump(2) << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
