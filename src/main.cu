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
    constexpr int width{ 3840 };
    constexpr int height{ 2160 };
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* const window{ SDL_CreateWindow("CUDA Software Rasteriser", 3840, SDL_WINDOWPOS_CENTERED, width, height, SDL_WindowFlags::SDL_WINDOW_FULLSCREEN) };
    SDL_Surface* const sdlSurface{ SDL_GetWindowSurface(window) };
    
    //Allocate pixel buffer
    uchar4* d_pixels;
    CC(cudaMalloc(&d_pixels, width * height * sizeof(uchar4)));
    renderDesc.surface = { width, height, d_pixels };
    
    
    //Create triangle data
    //Vertex buffer
    renderDesc.vertexBuffer.count = 4;
    renderDesc.vertexBuffer.size = sizeof(Vertex);
    renderDesc.vertexBuffer.stride = sizeof(Vertex);
    Vertex* h_vb{ static_cast<Vertex*>(malloc(renderDesc.vertexBuffer.count * sizeof(Vertex))) };
    h_vb[0] = { glm::vec3(-0.5f,  0.5f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f) }; //Bottom left (red)
    h_vb[1] = { glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f) }; //Top left (green)
    h_vb[2] = { glm::vec3( 0.5f,  0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f) }; //Bottom right (blue)
    h_vb[3] = { glm::vec3( 0.5f, -0.5f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f) }; //Top right (yellow)
    CC(cudaMalloc(&renderDesc.vertexBuffer.d_data, renderDesc.vertexBuffer.count * sizeof(Vertex)));
    CC(cudaMemcpy(renderDesc.vertexBuffer.d_data, h_vb, renderDesc.vertexBuffer.count * sizeof(Vertex), cudaMemcpyHostToDevice));
    renderDesc.vertexPositionAttributeIndex = 0;
    VertexLayout layout{};
    layout.attributes[0] = { 0, AttributeFormat::FLOAT3 };
    layout.attributes[1] = { sizeof(glm::vec3), AttributeFormat::FLOAT3 };
    renderDesc.vertexLayout = layout;
    
    //Index buffer (CW winding)
    renderDesc.indexBuffer.count = 6;
    renderDesc.indexBuffer.size = sizeof(std::uint32_t);
    renderDesc.indexBuffer.stride = sizeof(std::uint32_t);
    std::uint32_t* h_ib{ static_cast<std::uint32_t*>(malloc(renderDesc.indexBuffer.count * renderDesc.indexBuffer.size)) };
    h_ib[0] = 0;
    h_ib[1] = 1;
    h_ib[2] = 2;
    h_ib[3] = 2;
    h_ib[4] = 1;
    h_ib[5] = 3;
    CC(cudaMalloc(&renderDesc.indexBuffer.d_data, renderDesc.indexBuffer.count * renderDesc.indexBuffer.size));
    CC(cudaMemcpy(renderDesc.indexBuffer.d_data, h_ib, renderDesc.indexBuffer.count * renderDesc.indexBuffer.size, cudaMemcpyHostToDevice));
    
    
    glm::vec3 position{ 0,0,0 };
    float rotation{ 0.0f };
    constexpr float moveSpeed{ 4.0f };
    constexpr float rotSpeed{ 90.0f };
    
    Camera camera(45.0f, static_cast<float>(width) / static_cast<float>(height), 0.1f, 100.0f);
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
    free(h_vb);
    free(h_ib);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}