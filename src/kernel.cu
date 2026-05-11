#include "kernel.h"
#include <cstdint>
#include <iostream>


__global__ void RenderKernel(uchar4* const _pixels, const std::uint32_t _width, const std::uint32_t _height, const float2* const _d_vb)
{
    //Calculate the X and Y coordinates of the pixel this thread handles
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    
    if (x >= _width || y >= _height) { return; }
    
    //Calculate the 1D index into the pixel buffer
    const std::size_t idx{ y * _width + x };
    
    //Calculate the UV coordinate in screen space (range [-1, 1])
    const float aspectRatio{ static_cast<float>(_width) / static_cast<float>(_height) };
    const float u{ (static_cast<float>(x) / static_cast<float>(_width) * 2.0f - 1.0f) * aspectRatio };
    const float v{ (static_cast<float>(y) / static_cast<float>(_height)) * 2.0f - 1.0f };
    
    
    //Loop through all triangles in the vertex buffer and shade
    bool shaded{ false };
    for (std::size_t i{ 0 }; i < 6; i += 3)
    {
        //Calculate edge functions (2d cross products)
        const float2 p{ u, v };
        const float e0 = (_d_vb[i + 1].x - _d_vb[i + 0].x) * (p.y - _d_vb[i + 0].y) - (_d_vb[i + 1].y - _d_vb[i + 0].y) * (p.x - _d_vb[i + 0].x);
        const float e1 = (_d_vb[i + 2].x - _d_vb[i + 1].x) * (p.y - _d_vb[i + 1].y) - (_d_vb[i + 2].y - _d_vb[i + 1].y) * (p.x - _d_vb[i + 1].x);
        const float e2 = (_d_vb[i + 0].x - _d_vb[i + 2].x) * (p.y - _d_vb[i + 2].y) - (_d_vb[i + 0].y - _d_vb[i + 2].y) * (p.x - _d_vb[i + 2].x);
        
        //Point is inside triangle if all edge functions are >= 0 (CW winding)
        if (e0 >= 0 && e1 >= 0 && e2 >= 0)
        {
            //Calculate barycentric coordinates
            const float area{ e0 + e1 + e2 }; //area of the triangle
            if (area == 0.0f) { return; }
            const float w0{ e1 / area };
            const float w1{ e2 / area };
            const float w2{ e0 / area };
            
            //Interpolate colour based on barycentric weights
            const float r{ w0 * 255.0f };
            const float g{ w1 * 255.0f };
            const float b{ w2 * 255.0f };
            
            _pixels[idx].z = static_cast<unsigned char>(r);
            _pixels[idx].y = static_cast<unsigned char>(g);
            _pixels[idx].x = static_cast<unsigned char>(b);
            _pixels[idx].w = 255.0f;
            
            shaded = true;
        }
    }
    if (!shaded)
    {
        //Background colour (light blue)
        _pixels[idx].z = 160u;
        _pixels[idx].y = 230u;
        _pixels[idx].x = 255u;
        _pixels[idx].w = 255u;
    }
}


void Launch_RenderKernel(const dim3 _gridSize, const dim3 _blockSize, uchar4* const _pixels, const std::uint32_t _width, const std::uint32_t _height, const float2* const _d_vb)
{
    RenderKernel<<<_gridSize, _blockSize>>>(_pixels, _width, _height, _d_vb);
    
    const cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}