#pragma once

#include <cstddef>
#include <vector>

namespace cis565 {
namespace scan {

enum class Algorithm {
    Efficient,
};

struct Config {
    Algorithm algorithm = Algorithm::Efficient;
    int blockSize = 256;
    bool bankPadding = true;
};

// Persistent scratch for the renderer's hierarchical exclusive scan.  The
// constructor performs every allocation; scanDevice only enqueues kernels.
class ScanWorkspace {
public:
    explicit ScanWorkspace(std::size_t capacity, Config config = {});
    ~ScanWorkspace();

    ScanWorkspace(const ScanWorkspace&) = delete;
    ScanWorkspace& operator=(const ScanWorkspace&) = delete;
    ScanWorkspace(ScanWorkspace&&) = delete;
    ScanWorkspace& operator=(ScanWorkspace&&) = delete;

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t tileSize() const noexcept { return tileSize_; }
    const Config& config() const noexcept { return config_; }
    int* levelTotals(std::size_t level) noexcept { return totals_[level]; }
    int* levelOffsets(std::size_t level) noexcept { return offsets_[level]; }

private:
    std::size_t capacity_ = 0;
    std::size_t tileSize_ = 0;
    Config config_{};
    std::vector<int*> totals_;
    std::vector<int*> offsets_;
};

// Enqueue an exclusive scan on the default stream.  Input and output must be
// distinct device arrays, and n may not exceed workspace.capacity().
void scanDevice(std::size_t n, int* deviceOutput, const int* deviceInput,
    const Config& config, ScanWorkspace& workspace);

} // namespace scan
} // namespace cis565
