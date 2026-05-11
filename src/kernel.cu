#include "types.h"
#include "kernel.h"
#include <cstdint>
#include <iostream>


void CheckLaunchKernelSuccess()
{
    const cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}



__global__ void kClearSurface(const Surface* const _d_surface, const uchar4 _colour)
{
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    if (x >= _d_surface->width || y >= _d_surface->height) { return; }
    const std::size_t idx{ y * _d_surface->width + x };
    _d_surface->pixels[idx].z = _colour.x; //R
    _d_surface->pixels[idx].y = _colour.y; //G
    _d_surface->pixels[idx].x = _colour.z; //B
    _d_surface->pixels[idx].w = _colour.w; //A
}
void Launch_kClearSurface(const dim3 _gridSize, const dim3 _blockSize, const Surface* const _d_surface, const uchar4 _colour)
{
    kClearSurface<<<_gridSize, _blockSize>>>(_d_surface, _colour);
    CheckLaunchKernelSuccess();
}



__global__ void kRender(const Surface* const _surface, const std::uint32_t _width, const std::uint32_t _height, const float2* const _d_vb, const std::uint32_t* const _d_ib, const std::uint32_t _numIndices)
{
    //Calculate the X and Y coordinates of the pixel this thread handles
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    if (x >= _width || y >= _height) { return; }
    
    //Calculate the 1D index into the pixel buffer
    const std::size_t idx{ y * _width + x };
    
    //Calculate the UV coordinate in screen space (range [-1, 1])
    const float u{ (static_cast<float>(x) / static_cast<float>(_width) * 2.0f - 1.0f) };
    const float v{ (static_cast<float>(y) / static_cast<float>(_height)) * 2.0f - 1.0f };
    
    
    //Loop through all triangles in the vertex buffer and shade
    const float aspectRatio{ static_cast<float>(_width) / static_cast<float>(_height) };
    const float2 p{ u * aspectRatio, v };
    for (std::size_t i{ 0 }; i < _numIndices; i += 3)
    {
        //Calculate edge functions (2d cross products)
        const float e0 = (_d_vb[_d_ib[i + 1]].x - _d_vb[_d_ib[i + 0]].x) * (p.y - _d_vb[_d_ib[i + 0]].y) - (_d_vb[_d_ib[i + 1]].y - _d_vb[_d_ib[i + 0]].y) * (p.x - _d_vb[_d_ib[i + 0]].x);
        const float e1 = (_d_vb[_d_ib[i + 2]].x - _d_vb[_d_ib[i + 1]].x) * (p.y - _d_vb[_d_ib[i + 1]].y) - (_d_vb[_d_ib[i + 2]].y - _d_vb[_d_ib[i + 1]].y) * (p.x - _d_vb[_d_ib[i + 1]].x);
        const float e2 = (_d_vb[_d_ib[i + 0]].x - _d_vb[_d_ib[i + 2]].x) * (p.y - _d_vb[_d_ib[i + 2]].y) - (_d_vb[_d_ib[i + 0]].y - _d_vb[_d_ib[i + 2]].y) * (p.x - _d_vb[_d_ib[i + 2]].x);
        
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
            
            _surface->pixels[idx].z = static_cast<unsigned char>(r);
            _surface->pixels[idx].y = static_cast<unsigned char>(g);
            _surface->pixels[idx].x = static_cast<unsigned char>(b);
            _surface->pixels[idx].w = 255.0f;
        }
    }
}
void Launch_kRender(const dim3 _gridSize, const dim3 _blockSize, const Surface* const _surface, const std::uint32_t _width, const std::uint32_t _height, const float2* const _d_vb, const std::uint32_t* const _d_ib, const std::uint32_t _numIndices)
{
    kRender<<<_gridSize, _blockSize>>>(_surface, _width, _height, _d_vb, _d_ib, _numIndices);
    CheckLaunchKernelSuccess();
}



void GlobalSynchronise()
{
    cudaDeviceSynchronize();
}