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



__device__ std::uint32_t ReadBufferUint(const Buffer& _buffer, const std::size_t _index)
{
    const unsigned char* bytes{ static_cast<const unsigned char*>(_buffer.d_data) };
    const std::size_t byteOffset{ _index * _buffer.stride };
    if (_buffer.size == 1) //8-bit uint
    {
        return static_cast<std::uint32_t>(*reinterpret_cast<const std::uint8_t*>(bytes + byteOffset));
    }
    if (_buffer.size == 2) //16-bit uint
    {
        return static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(bytes + byteOffset));
    }
    if (_buffer.size == 4) //32-bit uint
    {
        return *reinterpret_cast<const std::uint32_t*>(bytes + byteOffset);
    }
    return 0u;
}



__device__ float ReadBufferFloat(const Buffer& _buffer, const std::size_t _index)
{
    const unsigned char* bytes{ static_cast<const unsigned char*>(_buffer.d_data) };
    const std::size_t byteOffset{ _index * _buffer.stride };
    if (_buffer.size == 4) //32-bit float
    {
        return *reinterpret_cast<const float*>(bytes + byteOffset);
    }
    return 0u;
}
__device__ void WriteBufferFloat(const Buffer& _buffer, const std::size_t _index, const float _val)
{
    unsigned char* bytes{ static_cast<unsigned char*>(_buffer.d_data) };
    const std::size_t byteOffset{ _index * _buffer.stride };
    if (_buffer.size == 4) //32-bit float
    {
        *reinterpret_cast<float*>(bytes + byteOffset) = _val;
    }
}



__device__ uchar4 ReadBufferUchar4(const Buffer& _buffer, const std::size_t _index)
{
    const unsigned char* bytes{ static_cast<const unsigned char*>(_buffer.d_data) };
    const std::size_t byteOffset{ _index * _buffer.stride };
    if (_buffer.size == 4)
    {
        return *reinterpret_cast<const uchar4*>(bytes + byteOffset);
    }
    return uchar4(0u,0u,0u,0u);
}



__device__ glm::vec4 ReadVertexAttribute(const Buffer& _buffer, const VertexLayout& _layout, std::size_t _vertexIndex, std::size_t _attrIndex)
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
        return glm::vec4{ v, 0.0f, 0.0f, 0.0f };
    }
    case AttributeFormat::FLOAT2:
    {
        const glm::vec2& v{ *reinterpret_cast<const glm::vec2*>(attrPtr) };
        return glm::vec4{ v.x, v.y, 0.0f, 0.0f };
    }
    case AttributeFormat::FLOAT3:
    {
        const float* f{ reinterpret_cast<const float*>(attrPtr) };
        return glm::vec4{ f[0], f[1], f[2], 0.0f };
    }
    case AttributeFormat::FLOAT4:
    {
        return *reinterpret_cast<const glm::vec4*>(attrPtr);
    }
    default:
        //std::cerr << "All formats in _desc.vertexLayout.attributes must be in {FLOAT, FLOAT2, FLOAT3, FLOAT4}. Format = " << static_cast<std::underlying_type_t<AttributeFormat>>(_layout.attributes[_attrIndex].format) << std::endl;
        //exit(-1);
        return glm::vec4(0,0,0,0);
    }
}



__global__ void kClearSurfaceAndDepthBuffer(const Surface& _surface, const Buffer& _depthBuffer, const uchar4 _colour, const float _depthValue)
{
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    if (x >= _surface.width || y >= _surface.height) { return; }
    const std::size_t idx{ y * _surface.width + x };
    
    _surface.d_pixels[idx].z = _colour.x; //R
    _surface.d_pixels[idx].y = _colour.y; //G
    _surface.d_pixels[idx].x = _colour.z; //B
    _surface.d_pixels[idx].w = _colour.w; //A
    
    WriteBufferFloat(_depthBuffer, idx, _depthValue);
}
void Launch_kClearSurfaceAndDepthBuffer(const dim3 _gridSize, const dim3 _blockSize, const Surface& _surface, const Buffer& _depthBuffer, const uchar4 _colour, const float _depthValue)
{
    kClearSurfaceAndDepthBuffer<<<_gridSize, _blockSize>>>(_surface, _depthBuffer, _colour, _depthValue);
    CheckLaunchKernelSuccess();
}



struct VertexShaderInput
{
    Buffer buffer;
    VertexLayout layout;
    std::size_t vertexIndex;
};

struct VertexShaderOutput
{
    glm::vec4 position;
    glm::vec4 colour;
    glm::vec2 uv;
};

__global__ void kVertexShader(const RenderDesc& _desc, VertexShaderOutput* d_vsOut)
{
    const std::uint32_t vertexIndex{ threadIdx.x + blockIdx.x * blockDim.x };
    if (vertexIndex >= _desc.vertexBuffer.count) { return; }

    //Read attributes and write to the output buffer
    glm::vec4 pos{ ReadVertexAttribute(_desc.vertexBuffer, _desc.vertexLayout, vertexIndex, 0) };
    pos.w = 1.0f;
    d_vsOut[vertexIndex].position = _desc.pushConstants.viewProjMatrix * _desc.pushConstants.modelMatrix * pos;
    d_vsOut[vertexIndex].colour = ReadVertexAttribute(_desc.vertexBuffer, _desc.vertexLayout, vertexIndex, 1);
    d_vsOut[vertexIndex].uv = ReadVertexAttribute(_desc.vertexBuffer, _desc.vertexLayout, vertexIndex, 2);
}



__device__ uchar4 kFragmentShader(const RenderDesc& _desc, const VertexShaderOutput& _input, const std::size_t _idx)
{
    // const unsigned char r{ static_cast<unsigned char>(_input.colour.z * 255.0f) };
    // const unsigned char g{ static_cast<unsigned char>(_input.colour.y * 255.0f) };
    // const unsigned char b{ static_cast<unsigned char>(_input.colour.x * 255.0f) };
    // return uchar4{ r, g, b, 255u };
    
    const int texX{ static_cast<int>(_input.uv.x * (_desc.textureWidth - 1)) };
    const int texY{ static_cast<int>(_input.uv.y * (_desc.textureHeight - 1)) };
    const std::size_t texIdx{ texY * _desc.textureWidth + texX };
    const uchar4 colour{ ReadBufferUchar4(_desc.texture, texIdx) };
    return { colour.z, colour.y, colour.x, colour.w };
}



__global__ void kRasterise(const RenderDesc& _desc, VertexShaderOutput* _d_vsOut)
{
    //Calculate the X and Y coordinates of the pixel this thread handles
    const std::uint32_t x{ threadIdx.x + blockIdx.x * blockDim.x };
    const std::uint32_t y{ threadIdx.y + blockIdx.y * blockDim.y };
    if (x >= _desc.surface.width || y >= _desc.surface.height) { return; }
    
    //Calculate the 1D index into the pixel buffer
    const std::size_t idx{ y * _desc.surface.width + x };
    
    //Calculate the UV coordinate in clip space (range [-1, 1])
    const float u{ (static_cast<float>(x) / static_cast<float>(_desc.surface.width) * 2.0f - 1.0f) };
    const float v{ (1.0f - static_cast<float>(y) / static_cast<float>(_desc.surface.height) * 2.0f) };
    
    
    //Loop through all indices in the index buffer and shade triangles
    const float2 p{ u, v };

    for (std::size_t i{ 0 }; i < _desc.indexBuffer.count; i += 3)
    {
        //Get vertex positions
        const std::uint32_t idx0{ ReadBufferUint(_desc.indexBuffer, i + 0) };
        const std::uint32_t idx1{ ReadBufferUint(_desc.indexBuffer, i + 1) };
        const std::uint32_t idx2{ ReadBufferUint(_desc.indexBuffer, i + 2) };
        
        const float w0{ _d_vsOut[idx0].position.w };
        const float w1{ _d_vsOut[idx1].position.w };
        const float w2{ _d_vsOut[idx2].position.w };
        if (w0 <= 0.0f || w1 <= 0.0f || w2 <= 0.0f) { continue; }
        
        const float3 v0{ _d_vsOut[idx0].position.x / w0, _d_vsOut[idx0].position.y / w0, _d_vsOut[idx0].position.z / w0 };
        const float3 v1{ _d_vsOut[idx1].position.x / w1, _d_vsOut[idx1].position.y / w1, _d_vsOut[idx1].position.z / w1 };
        const float3 v2{ _d_vsOut[idx2].position.x / w2, _d_vsOut[idx2].position.y / w2, _d_vsOut[idx2].position.z / w2 };
        
        //Calculate edge functions (2d cross products)
        const float e0{ (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x) };
        const float e1{ (v2.x - v1.x) * (p.y - v1.y) - (v2.y - v1.y) * (p.x - v1.x) };
        const float e2{ (v0.x - v2.x) * (p.y - v2.y) - (v0.y - v2.y) * (p.x - v2.x) };
        
        //Point is inside triangle if all edge functions are <= 0 (CW winding)
        if (e0 <= 0 && e1 <= 0 && e2 <= 0)
        {
            //Calculate barycentric coordinates
            const float area{ e0 + e1 + e2 }; //area of the triangle
            if (area == 0.0f) { continue; }
            const float w0_bary{ e1 / area };
            const float w1_bary{ e2 / area };
            const float w2_bary{ e0 / area };
            
            //Barycentric interpolation
            VertexShaderOutput interpolated;
            interpolated.position.x = w0_bary * _d_vsOut[idx0].position.x + w1_bary * _d_vsOut[idx1].position.x + w2_bary * _d_vsOut[idx2].position.x;
            interpolated.position.y = w0_bary * _d_vsOut[idx0].position.y + w1_bary * _d_vsOut[idx1].position.y + w2_bary * _d_vsOut[idx2].position.y;
            interpolated.position.z = w0_bary * _d_vsOut[idx0].position.z + w1_bary * _d_vsOut[idx1].position.z + w2_bary * _d_vsOut[idx2].position.z;
            interpolated.position.w = 1.0f;

            //Early depth test
            const float depth{ w0_bary * v0.z + w1_bary * v1.z + w2_bary * v2.z };
            if (depth < ReadBufferFloat(_desc.depthBuffer, idx))
            {
                WriteBufferFloat(_desc.depthBuffer, idx, depth);
    
                const float oneOverW0 = 1.0f / w0;
                const float oneOverW1 = 1.0f / w1;
                const float oneOverW2 = 1.0f / w2;
    
                const float oneOverW = w0_bary * oneOverW0 + w1_bary * oneOverW1 + w2_bary * oneOverW2;
                const float w = 1.0f / oneOverW;
    
                interpolated.uv.x = (w0_bary * _d_vsOut[idx0].uv.x * oneOverW0 + 
                                     w1_bary * _d_vsOut[idx1].uv.x * oneOverW1 + 
                                     w2_bary * _d_vsOut[idx2].uv.x * oneOverW2) * w;
                interpolated.uv.y = (w0_bary * _d_vsOut[idx0].uv.y * oneOverW0 + 
                                     w1_bary * _d_vsOut[idx1].uv.y * oneOverW1 + 
                                     w2_bary * _d_vsOut[idx2].uv.y * oneOverW2) * w;
    
                interpolated.colour.x = (w0_bary * _d_vsOut[idx0].colour.x * oneOverW0 + 
                                         w1_bary * _d_vsOut[idx1].colour.x * oneOverW1 + 
                                         w2_bary * _d_vsOut[idx2].colour.x * oneOverW2) * w;
                interpolated.colour.y = (w0_bary * _d_vsOut[idx0].colour.y * oneOverW0 + 
                                         w1_bary * _d_vsOut[idx1].colour.y * oneOverW1 + 
                                         w2_bary * _d_vsOut[idx2].colour.y * oneOverW2) * w;
                interpolated.colour.z = (w0_bary * _d_vsOut[idx0].colour.z * oneOverW0 + 
                                         w1_bary * _d_vsOut[idx1].colour.z * oneOverW1 + 
                                         w2_bary * _d_vsOut[idx2].colour.z * oneOverW2) * w;
                interpolated.colour.w = (w0_bary * _d_vsOut[idx0].colour.w * oneOverW0 + 
                                         w1_bary * _d_vsOut[idx1].colour.w * oneOverW1 + 
                                         w2_bary * _d_vsOut[idx2].colour.w * oneOverW2) * w;
                
                _desc.surface.d_pixels[idx] = kFragmentShader(_desc, interpolated, idx);
            }
        }
    }
}



void Render(const RenderDesc& _desc)
{
    VertexShaderOutput* d_vsOut;
    cudaMalloc(&d_vsOut, _desc.vertexBuffer.count * sizeof(VertexShaderOutput));
    {
        dim3 blockSize(256, 1, 1);
        dim3 gridSize((_desc.vertexBuffer.count + blockSize.x - 1) / blockSize.x);
        kVertexShader<<<gridSize, blockSize>>>(_desc, d_vsOut);
        CheckLaunchKernelSuccess();
    }
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((_desc.surface.width + blockSize.x - 1) / blockSize.x, (_desc.surface.height + blockSize.y - 1) / blockSize.y);
        kRasterise<<<gridSize, blockSize>>>(_desc, d_vsOut);
        CheckLaunchKernelSuccess();
    }
    cudaFree(d_vsOut);
}



void GlobalSynchronise()
{
    cudaDeviceSynchronize();
    CheckLaunchKernelSuccess();
}