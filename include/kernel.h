#pragma once


#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


void Launch_kClearSurface(dim3 _gridSize, dim3 _blockSize, Surface _surface, uchar4 _colour);
void Render(const RenderDesc& _desc);
void GlobalSynchronise();