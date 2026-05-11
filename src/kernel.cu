#include "kernel.h"
#include <cstdint>
#include <iostream>


__global__ void RenderKernel(uchar4* const _pixels, const std::uint32_t _width, const std::uint32_t _height, const float2* const _d_tri)
{
    //Calculate the X and Y coordinates of the pixel this thread handles
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    
    if (x >= _width || y >= _height) { return; }
    
    //Calculate the 1D index into the pixel buffer
    const std::size_t idx{ y * _width + x };
    
    //Calculate the UV coordinate in screen space (range [-1, 1])
    const float u{ (static_cast<float>(x) / static_cast<float>(_width)) * 2.0f - 1.0f };
    const float v{ (static_cast<float>(y) / static_cast<float>(_height)) * 2.0f - 1.0f };
    
    //Calculate edge functions (2d cross products)
    const float2 p{ u, v };
    const float e0 = (_d_tri[1].x - _d_tri[0].x) * (p.y - _d_tri[0].y) - (_d_tri[1].y - _d_tri[0].y) * (p.x - _d_tri[0].x);
    const float e1 = (_d_tri[2].x - _d_tri[1].x) * (p.y - _d_tri[1].y) - (_d_tri[2].y - _d_tri[1].y) * (p.x - _d_tri[1].x);
    const float e2 = (_d_tri[0].x - _d_tri[2].x) * (p.y - _d_tri[2].y) - (_d_tri[0].y - _d_tri[2].y) * (p.x - _d_tri[2].x);
    
    //Point is inside triangle if all edge functions are >= 0 (CW winding)
    if (e0 >= 0 && e1 >= 0 && e2 >= 0)
    {
        _pixels[idx].z = 255u;
    }
}


void Launch_RenderKernel(const dim3 _gridSize, const dim3 _blockSize, uchar4* const _pixels, const std::uint32_t _width, const std::uint32_t _height, const float2* const _d_tri)
{
    RenderKernel<<<_gridSize, _blockSize>>>(_pixels, _width, _height, _d_tri);
    
    const cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}