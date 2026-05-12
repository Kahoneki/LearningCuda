#pragma once


#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


void Launch_kClearSurfaceAndDepthBuffer(dim3 _gridSize, dim3 _blockSize, const Surface& _surface, const Buffer& _depthBuffer, uchar4 _colour, float _depthValue);
void Render(const RenderDesc& _desc);
void GlobalSynchronise();