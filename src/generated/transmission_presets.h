#pragma once

// CPU lookup thats like metal_presets.h. 
//  glass surrounded by air.
struct TransmissionIORPreset {
    float ior;
};

inline constexpr TransmissionIORPreset transmissionIORPresets[] = {
    {1.5f} // 0: Glass
};
