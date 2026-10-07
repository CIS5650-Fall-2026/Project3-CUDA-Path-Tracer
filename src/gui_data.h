#pragma once

#include <string>

class GuiDataContainer {
  public:
    GuiDataContainer() : traced_depth(0) {}
    int traced_depth;
    bool load_requested = false;
    bool env_requested = false;
    std::string env_path;
};
