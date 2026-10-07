#pragma once

inline constexpr float PI = 3.1415926535897932384626422832795028841971f;
inline constexpr float INV_PI = 0.3183098861837906715377675267450287240689f;
inline constexpr float INV_FOUR_PI = 0.07957747154594766788f;
inline constexpr float TWO_PI = 6.2831853071795864769252867665590057683943f;
inline constexpr float SQRT_OF_ONE_THIRD = 0.5773502691896257645091487805019574556476f;

namespace math_utils {
inline int divup(int x, int n) {
    return (x + n - 1) / n;
}
inline float clamp(float f, float min, float max) {
    if (f < min) {
        return min;
    } else if (f > max) {
        return max;
    } else {
        return f;
    }
}
} // namespace math_utils
