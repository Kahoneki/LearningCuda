#include "kernel.h"
#include <cstdint>
#include <iostream>


__global__ void RenderKernel(uchar4* const _pixels, const std::uint32_t _width, const std::uint32_t _height)
{
    //Calculate the X and Y coordinates of the pixel this thread handles
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    
    if (x >= _width || y >= _height) { return; }
    
    //Calculate the 1D index into the pixel buffer
    const std::size_t idx{ y * _width + x };
    
    //rasteriser will go here eventually
    _pixels[idx].z = static_cast<unsigned char>(x * 255 / _width); //Red
    _pixels[idx].y = static_cast<unsigned char>(y * 255 / _height); //Green
    _pixels[idx].x = 0u; //Blue
    _pixels[idx].w = 255u; //Alpha
}


void Launch_RenderKernel(const dim3 _gridSize, const dim3 _blockSize, uchar4* const _pixels, const std::uint32_t _width, const std::uint32_t _height)
{
    RenderKernel<<<_gridSize, _blockSize>>>(_pixels, _width, _height);
    
    const cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}