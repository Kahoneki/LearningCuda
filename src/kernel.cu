#include "kernel.h"
#include <cstdint>
#include <iostream>


__global__ void vector_add_kernel(const float* _a, const float* _b, float* _c, const std::uint32_t _n)
{
    const std::uint32_t idx{ threadIdx.x + blockIdx.x * blockDim.x };
    if (idx < _n)
    {
        for (std::size_t i{ 0 }; i < 30000; ++i)
        {
            _c[idx] = _a[idx] + _b[idx];
        }
    }
}



void launch_vector_add(const float* _a, const float* _b, float* _c, const std::uint32_t _n)
{
    constexpr std::uint32_t threads_per_block{ 256 };
    std::uint32_t blocks{ (_n + threads_per_block - 1) / threads_per_block };
    
    vector_add_kernel<<<blocks, threads_per_block>>>(_a,_b,_c,_n);
    
    const cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}
