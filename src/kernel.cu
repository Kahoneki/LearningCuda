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



__device__ std::uint32_t ReadIndex(const Buffer& _buffer, const std::size_t _index)
{
    const unsigned char* bytes{ static_cast<const unsigned char*>(_buffer.d_data) };
    const std::size_t byteOffset{ _index * _buffer.stride };
    if (_buffer.size == 1) //8-bit indices
    {
        return static_cast<std::uint32_t>(*reinterpret_cast<const std::uint8_t*>(bytes + byteOffset));
    }
    if (_buffer.size == 2) //16-bit indices
    {
        return static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(bytes + byteOffset));
    }
    if (_buffer.size == 4) //32-bit indices
    {
        return *reinterpret_cast<const std::uint32_t*>(bytes + byteOffset);
    }
    //std::cerr << "_desc.indexBuffer.size must be in {1, 2, 4}. Size = " << _buffer.size << std::endl;
    //exit(-1);
    return 0u;
}



__device__ float4 ReadVertexAttribute(const Buffer& _buffer, const VertexLayout& _layout, std::size_t _vertexIndex, std::size_t _attrIndex)
{
    const unsigned char* bytes = static_cast<const unsigned char*>(_buffer.d_data);
    const std::size_t vertexByteOffset = _vertexIndex * _buffer.stride;
    const std::size_t finalOffset = vertexByteOffset + _layout.attributes[_attrIndex].offset;
    const unsigned char* attrPtr = bytes + finalOffset;

    switch (_layout.attributes[_attrIndex].format)
    {
    case AttributeFormat::FLOAT:
    {
        const float v{ *reinterpret_cast<const float*>(attrPtr) };
        return float4{ v, 0.0f, 0.0f, 0.0f };
    }
    case AttributeFormat::FLOAT2:
    {
        const float2& v{ *reinterpret_cast<const float2*>(attrPtr) };
        return float4{ v.x, v.y, 0.0f, 0.0f };
    }
    case AttributeFormat::FLOAT3:
    {
        const float* f{ reinterpret_cast<const float*>(attrPtr) };
        return float4{ f[0], f[1], f[2], 0.0f };
    }
    case AttributeFormat::FLOAT4:
    {
        return *reinterpret_cast<const float4*>(attrPtr);
    }
    default:
        //std::cerr << "All formats in _desc.vertexLayout.attributes must be in {FLOAT, FLOAT2, FLOAT3, FLOAT4}. Format = " << static_cast<std::underlying_type_t<AttributeFormat>>(_layout.attributes[_attrIndex].format) << std::endl;
        //exit(-1);
        return float4(0,0,0,0);
    }
}



__global__ void kClearSurface(const Surface _surface, const uchar4 _colour)
{
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    if (x >= _surface.width || y >= _surface.height) { return; }
    const std::size_t idx{ y * _surface.width + x };
    _surface.d_pixels[idx].z = _colour.x; //R
    _surface.d_pixels[idx].y = _colour.y; //G
    _surface.d_pixels[idx].x = _colour.z; //B
    _surface.d_pixels[idx].w = _colour.w; //A
}
void Launch_kClearSurface(const dim3 _gridSize, const dim3 _blockSize, const Surface _surface, const uchar4 _colour)
{
    kClearSurface<<<_gridSize, _blockSize>>>(_surface, _colour);
    CheckLaunchKernelSuccess();
}



__global__ void kRender(const RenderDesc& _desc)
{
    //Calculate the X and Y coordinates of the pixel this thread handles
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    if (x >= _desc.surface.width || y >= _desc.surface.height) { return; }
    
    //Calculate the 1D index into the pixel buffer
    const std::size_t idx{ y * _desc.surface.width + x };
    
    //Calculate the UV coordinate in screen space (range [-1, 1])
    const float u{ (static_cast<float>(x) / static_cast<float>(_desc.surface.width) * 2.0f - 1.0f) };
    const float v{ (static_cast<float>(y) / static_cast<float>(_desc.surface.height) * 2.0f - 1.0f) };
    
    
    //Loop through all indices in the index buffer and shade triangles
    const float aspectRatio{ static_cast<float>(_desc.surface.width) / static_cast<float>(_desc.surface.height) };
    const float2 p{ u * aspectRatio, v };
    for (std::size_t i{ 0 }; i < _desc.indexBuffer.count; i += 3)
    {
        //Get vertex positions
        const std::uint32_t idx0{ ReadIndex(_desc.indexBuffer, i + 0) };
        const std::uint32_t idx1{ ReadIndex(_desc.indexBuffer, i + 1) };
        const std::uint32_t idx2{ ReadIndex(_desc.indexBuffer, i + 2) };
        const float4 v0_f4{ ReadVertexAttribute(_desc.vertexBuffer, _desc.vertexLayout, idx0, _desc.vertexPositionAttributeIndex) };
        const float4 v1_f4{ ReadVertexAttribute(_desc.vertexBuffer, _desc.vertexLayout, idx1, _desc.vertexPositionAttributeIndex) };
        const float4 v2_f4{ ReadVertexAttribute(_desc.vertexBuffer, _desc.vertexLayout, idx2, _desc.vertexPositionAttributeIndex) };
        const float2 v0{ v0_f4.x, v0_f4.y };
        const float2 v1{ v1_f4.x, v1_f4.y };
        const float2 v2{ v2_f4.x, v2_f4.y };
        
        //Calculate edge functions (2d cross products)
        const float e0{ (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x) };
        const float e1{ (v2.x - v1.x) * (p.y - v1.y) - (v2.y - v1.y) * (p.x - v1.x) };
        const float e2{ (v0.x - v2.x) * (p.y - v2.y) - (v0.y - v2.y) * (p.x - v2.x) };
        
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
            
            _desc.surface.d_pixels[idx].z = static_cast<unsigned char>(r);
            _desc.surface.d_pixels[idx].y = static_cast<unsigned char>(g);
            _desc.surface.d_pixels[idx].x = static_cast<unsigned char>(b);
            _desc.surface.d_pixels[idx].w = 255.0f;
        }
    }
}
void Launch_kRender(const dim3 _gridSize, const dim3 _blockSize, const RenderDesc& _desc)
{
    kRender<<<_gridSize, _blockSize>>>(_desc);
    CheckLaunchKernelSuccess();
}



void GlobalSynchronise()
{
    cudaDeviceSynchronize();
    CheckLaunchKernelSuccess();
}