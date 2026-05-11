#pragma once
#include <cstdint>


#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void Launch_kClearSurface(dim3 _gridSize, dim3 _blockSize, const Surface* _d_surface, uchar4 _colour);
void Launch_kRender(dim3 _gridSize, dim3 _blockSize, const Surface* _surface, std::uint32_t _width, std::uint32_t _height, const float2* _d_vb, const std::uint32_t* _d_ib, std::uint32_t _numIndices);
void GlobalSynchronise();