#pragma once
#include <cstdint>

struct Surface
{
    Surface(const std::uint32_t _width, const std::uint32_t _height, uchar4* const _pixels) : width(_width), height(_height), pixels(_pixels) {}
    std::uint32_t width;
    std::uint32_t height;
    uchar4* pixels;
};
