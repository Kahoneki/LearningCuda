#pragma once
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdint>


struct Surface
{
    Surface(const std::uint32_t _width, const std::uint32_t _height, uchar4* const _d_pixels) : width(_width), height(_height), d_pixels(_d_pixels) {}
    Surface() {}
    std::uint32_t width{ 0 };
    std::uint32_t height{ 0 };
    uchar4* d_pixels{ nullptr };
};



struct Buffer
{
    //Data
    void* d_data;
    
    //Size of each value in bytes
    std::size_t size;
    
    //Number of values
    std::size_t count;
    
    //Distance between values in bytes (for tightly packed data, Buffer.stride = Buffer.size)
    std::size_t stride;
};


enum class AttributeFormat : std::uint8_t
{
    FLOAT,
    FLOAT2,
    FLOAT3,
    FLOAT4,
};


struct VertexAttributeDesc
{
    std::uint32_t offset;
    AttributeFormat format;
};


using AttrIndex_t = std::uint8_t;
struct VertexLayout
{
    static constexpr AttrIndex_t MAX_ATTRIBUTES{ 4 };
    VertexAttributeDesc attributes[MAX_ATTRIBUTES]{};
};


struct PushConstants
{
    glm::mat4 modelMatrix;
    glm::mat4 viewProjMatrix;
};


struct RenderDesc
{
    Surface surface;
    
    Buffer vertexBuffer;
    VertexLayout vertexLayout;
    AttrIndex_t vertexPositionAttributeIndex;
    Buffer indexBuffer;
    
    Buffer depthBuffer;
    
    Buffer texture;
    std::uint32_t textureWidth;
    std::uint32_t textureHeight;
    
    PushConstants pushConstants;
};