#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void launch_vector_add(const float* _a, const float* _b, float* _c, std::uint32_t _n);