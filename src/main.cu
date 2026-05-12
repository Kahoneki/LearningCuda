#include "Camera.h"
#include "types.h"
#include "kernel.h"
#include <SDL2/SDL.h>
#include <iostream>


#define CUDA_CHECK(call)                                                                                                    \
    do                                                                                                                      \
    {                                                                                                                       \
        const cudaError_t err{ call };                                                                                      \
        if (err != cudaSuccess)                                                                                             \
        {                                                                                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(err);                                                                                                      \
        }                                                                                                                   \
    } while (0)

#define CC(call) CUDA_CHECK(call)


struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
};


int main()
{
    RenderDesc renderDesc{};
    
    
    //Initialise SDL window and surface
    constexpr int width{ 1920 };
    constexpr int height{ 1080 };
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* const window{ SDL_CreateWindow("CUDA Software Rasteriser", 3840, SDL_WINDOWPOS_CENTERED, width, height, 0) };
    SDL_Surface* const sdlSurface{ SDL_GetWindowSurface(window) };
    
    //Allocate pixel buffer
    uchar4* d_pixels;
    CC(cudaMalloc(&d_pixels, width * height * sizeof(uchar4)));
    renderDesc.surface = { width, height, d_pixels };
    
    
    //Create triangle data
    //Vertex buffer
    renderDesc.vertexBuffer.count = 24;
    renderDesc.vertexBuffer.size = sizeof(Vertex);
    renderDesc.vertexBuffer.stride = sizeof(Vertex);
    Vertex h_vb[24]
    {
        //--- Front Face (Z = +0.5) - Red ---
        { glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(1.0f, 0.0f, 0.0f) }, //0
        { glm::vec3( 0.5f, -0.5f,  0.5f), glm::vec3(1.0f, 0.0f, 0.0f) }, //1
        { glm::vec3( 0.5f,  0.5f,  0.5f), glm::vec3(1.0f, 0.0f, 0.0f) }, //2
        { glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(1.0f, 0.0f, 0.0f) }, //3

        //--- Back Face (Z = -0.5) - Green ---
        { glm::vec3( 0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 0.0f) }, //4
        { glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 0.0f) }, //5
        { glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 0.0f) }, //6
        { glm::vec3( 0.5f,  0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 0.0f) }, //7

        //--- Top Face (Y = +0.5) - Blue ---
        { glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(0.0f, 0.0f, 1.0f) }, //8
        { glm::vec3( 0.5f,  0.5f,  0.5f), glm::vec3(0.0f, 0.0f, 1.0f) }, //9
        { glm::vec3( 0.5f,  0.5f, -0.5f), glm::vec3(0.0f, 0.0f, 1.0f) }, //10
        { glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.0f, 0.0f, 1.0f) }, //11

        //--- Bottom Face (Y = -0.5) - Yellow ---
        { glm::vec3( 0.5f, -0.5f,  0.5f), glm::vec3(1.0f, 1.0f, 0.0f) }, //12
        { glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(1.0f, 1.0f, 0.0f) }, //13
        { glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(1.0f, 1.0f, 0.0f) }, //14
        { glm::vec3( 0.5f, -0.5f, -0.5f), glm::vec3(1.0f, 1.0f, 0.0f) }, //15

        //--- Right Face (X = +0.5) - Magenta ---
        { glm::vec3( 0.5f, -0.5f,  0.5f), glm::vec3(1.0f, 0.0f, 1.0f) }, //16
        { glm::vec3( 0.5f, -0.5f, -0.5f), glm::vec3(1.0f, 0.0f, 1.0f) }, //17
        { glm::vec3( 0.5f,  0.5f, -0.5f), glm::vec3(1.0f, 0.0f, 1.0f) }, //18
        { glm::vec3( 0.5f,  0.5f,  0.5f), glm::vec3(1.0f, 0.0f, 1.0f) }, //19

        //--- Left Face (X = -0.5) - Cyan ---
        { glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 1.0f) }, //20
        { glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(0.0f, 1.0f, 1.0f) }, //21
        { glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(0.0f, 1.0f, 1.0f) }, //22
        { glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.0f, 1.0f, 1.0f) }  //23
    };
    CC(cudaMalloc(&renderDesc.vertexBuffer.d_data, renderDesc.vertexBuffer.count * sizeof(Vertex)));
    CC(cudaMemcpy(renderDesc.vertexBuffer.d_data, h_vb, renderDesc.vertexBuffer.count * sizeof(Vertex), cudaMemcpyHostToDevice));
    renderDesc.vertexPositionAttributeIndex = 0;
    VertexLayout layout{};
    layout.attributes[0] = { 0, AttributeFormat::FLOAT3 };
    layout.attributes[1] = { sizeof(glm::vec3), AttributeFormat::FLOAT3 };
    renderDesc.vertexLayout = layout;
    
    //Index buffer (CW winding)
    renderDesc.indexBuffer.count = 36;
    renderDesc.indexBuffer.size = sizeof(std::uint32_t);
    renderDesc.indexBuffer.stride = sizeof(std::uint32_t);
    std::uint32_t h_ib[36]
    {
        //Front
        0,  1,  2,  0,  2,  3,
        //Back
        4,  5,  6,  4,  6,  7,
        //Top
        8,  9, 10,  8, 10, 11,
        //Bottom
        12, 13, 14, 12, 14, 15,
        //Right
        16, 17, 18, 16, 18, 19,
        //Left
        20, 21, 22, 20, 22, 23
    };
    CC(cudaMalloc(&renderDesc.indexBuffer.d_data, renderDesc.indexBuffer.count * renderDesc.indexBuffer.size));
    CC(cudaMemcpy(renderDesc.indexBuffer.d_data, h_ib, renderDesc.indexBuffer.count * renderDesc.indexBuffer.size, cudaMemcpyHostToDevice));
    
    
    glm::vec3 position{ 0,0,0 };
    float rotation{ 0.0f };
    constexpr float moveSpeed{ 4.0f };
    constexpr float rotSpeed{ 90.0f };
    
    Camera camera(45.0f, static_cast<float>(width) / static_cast<float>(height), 0.01f, 100.0f);
    Uint64 lastTime{ SDL_GetPerformanceCounter() };
    Uint64 frequency{ SDL_GetPerformanceFrequency() };
    
    //Main loop
    bool running{ true };
    SDL_Event e;
    while (running)
    {
        const Uint64 currentTime{ SDL_GetPerformanceCounter() };
        const float dt{ static_cast<float>(currentTime - lastTime) / static_cast<float>(frequency) };
        lastTime = currentTime;
        
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT) { running = false; }
            camera.HandleEvent(e);
        }
        camera.Update(dt);
        
        position.x = sin(moveSpeed * currentTime) / 2.0f;
        rotation = rotSpeed * currentTime;
        glm::mat4 modelMat{ glm::mat4(1.0f) };
        // modelMat = glm::translate(modelMat, position);
        // modelMat = glm::rotate(modelMat, glm::radians(rotation), glm::vec3(0,0,1));
        renderDesc.pushConstants.modelMatrix = modelMat;
        renderDesc.pushConstants.viewProjMatrix = camera.GetProjMatrix() * camera.GetViewMatrix();
        
        //Calculate kernel parameters
        constexpr dim3 blockSize(16, 16);
        const dim3 gridSize((renderDesc.surface.width + blockSize.x - 1) / blockSize.x, (renderDesc.surface.height + blockSize.y - 1) / blockSize.y);
        
        //Render
        Launch_kClearSurface(gridSize, blockSize, renderDesc.surface, { 160, 230, 255, 255 });
        Render(renderDesc);
        GlobalSynchronise();
        
        //Copy pixels from GPU to CPU
        //Lock the surface so SDL doesn't overwrite it while we write
        SDL_LockSurface(sdlSurface);
        CC(cudaMemcpy(sdlSurface->pixels, d_pixels, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
        SDL_UnlockSurface(sdlSurface);
        
        //Push CPU pixels to the monitor
        SDL_UpdateWindowSurface(window);
    }
    
    
    //Cleanup
    CC(cudaFree(d_pixels));
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}