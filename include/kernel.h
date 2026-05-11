#pragma once
#include <cstdint>


#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void Launch_RenderKernel(dim3 _gridSize, dim3 _blockSize, uchar4* _pixels, std::uint32_t _width, std::uint32_t _height, const float2* _d_tri);