// Original verification wrapper. PNG decoding is supplied by the unchanged
// approved starter header; no external decoder implementation is copied here.
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

struct ImageFree {
    void operator()(unsigned char* value) const { stbi_image_free(value); }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: interaction_pixel_compare first.png second.png\n";
        return 2;
    }
    int aw = 0, ah = 0, ac = 0, bw = 0, bh = 0, bc = 0;
    std::unique_ptr<unsigned char, ImageFree> a(
        stbi_load(argv[1], &aw, &ah, &ac, 3));
    if (!a) {
        std::cerr << "first PNG decode failed: " << stbi_failure_reason() << '\n';
        return 2;
    }
    std::unique_ptr<unsigned char, ImageFree> b(
        stbi_load(argv[2], &bw, &bh, &bc, 3));
    if (!b) {
        std::cerr << "second PNG decode failed: " << stbi_failure_reason() << '\n';
        return 2;
    }
    if (aw != bw || ah != bh || aw <= 0 || ah <= 0 || ac != 3 || bc != 3) {
        std::cerr << "images must have matching positive dimensions and RGB source channels\n";
        return 2;
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(aw) * ah;
    std::uint64_t changedPixels = 0, changedChannels = 0, absoluteSum = 0;
    int maximum = 0;
    std::vector<std::uint64_t> examples;
    for (std::uint64_t p = 0; p < pixels; ++p) {
        bool changed = false;
        for (int c = 0; c < 3; ++c) {
            const std::uint64_t i = 3 * p + c;
            const int difference = std::abs(static_cast<int>(a.get()[i])
                - static_cast<int>(b.get()[i]));
            maximum = std::max(maximum, difference);
            absoluteSum += difference;
            if (difference != 0) {
                changed = true;
                ++changedChannels;
                if (examples.size() < 12) examples.push_back(i);
            }
        }
        changedPixels += changed ? 1 : 0;
    }
    const bool passed = maximum <= 1;
    std::cout << "{\"passed\":" << (passed ? "true" : "false")
        << ",\"width\":" << aw << ",\"height\":" << ah
        << ",\"channels\":3,\"total_pixels\":" << pixels
        << ",\"differing_pixels\":" << changedPixels
        << ",\"differing_channels\":" << changedChannels
        << ",\"maximum_absolute_channel_difference\":" << maximum
        << ",\"sum_absolute_channel_differences\":" << absoluteSum
        << ",\"rgb8_equal\":" << (maximum == 0 ? "true" : "false")
        << ",\"acceptance_maximum_absolute_channel_difference\":1"
        << ",\"first_differences\":[";
    for (std::size_t j = 0; j < examples.size(); ++j) {
        const std::uint64_t i = examples[j];
        const std::uint64_t p = i / 3;
        if (j != 0) std::cout << ',';
        std::cout << "{\"x\":" << p % aw << ",\"y\":" << p / aw
            << ",\"channel\":" << i % 3
            << ",\"preview\":" << static_cast<int>(a.get()[i])
            << ",\"reference\":" << static_cast<int>(b.get()[i]) << '}';
    }
    std::cout << "]}\n";
    return passed ? 0 : 1;
}
