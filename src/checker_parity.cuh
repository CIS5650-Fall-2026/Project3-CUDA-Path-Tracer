#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define CHECKER_PARITY_HD __host__ __device__
#define CHECKER_PARITY_INLINE __forceinline__
#else
#define CHECKER_PARITY_HD
#define CHECKER_PARITY_INLINE inline
#endif

namespace renderer
{
namespace checker_parity_detail
{

CHECKER_PARITY_HD CHECKER_PARITY_INLINE bool finiteFloat(float value)
{
    return value == value && value >= -FLT_MAX && value <= FLT_MAX;
}

CHECKER_PARITY_HD CHECKER_PARITY_INLINE unsigned int cellParity(
    float coordinate, float scale)
{
    const float product = coordinate * scale;
    if (finiteFloat(product))
    {
        const float cell = floorf(product);
        if (cell >= -2147483648.0f && cell < 2147483648.0f)
        {
            const std::int32_t signedCell = static_cast<std::int32_t>(cell);
            return static_cast<std::uint32_t>(signedCell) & 1u;
        }

        return fmodf(fabsf(cell), 2.0f) == 0.0f ? 0u : 1u;
    }

    // A product of two finite floats always fits in double. This branch only
    // defines parity for the case where the original float multiply overflowed.
    const double wideProduct = static_cast<double>(coordinate)
        * static_cast<double>(scale);
    const double cell = floor(wideProduct);
    return fmod(fabs(cell), 2.0) == 0.0 ? 0u : 1u;
}

} // namespace checker_parity_detail

CHECKER_PARITY_HD CHECKER_PARITY_INLINE unsigned int checkerParity(
    float u, float v, float scale)
{
    using namespace checker_parity_detail;

    if (!finiteFloat(u) || !finiteFloat(v) || !finiteFloat(scale))
    {
        return 0u;
    }

    return cellParity(u, scale) ^ cellParity(v, scale);
}

} // namespace renderer

#undef CHECKER_PARITY_INLINE
#undef CHECKER_PARITY_HD
