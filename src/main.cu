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
            exit(EXIT_FAILURE);                                                                                             \
        }                                                                                                                   \
    } while (0)

#define CC(call) CUDA_CHECK(call)


int main()
{
    //Initialise SDL window and surface
    constexpr int width{ 1920 };
    constexpr int height{ 1080 };
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* const window{ SDL_CreateWindow("CUDA Software Rasteriser", 3840, SDL_WINDOWPOS_CENTERED, width, height, 0) };
    SDL_Surface* const sdlSurface{ SDL_GetWindowSurface(window) };
    
    //Allocate pixel buffer
    uchar4* d_pixels;
    CC(cudaMalloc(&d_pixels, width * height * sizeof(uchar4)));
    const Surface surface{ width, height, d_pixels };
    
    
    //Create triangle data
    //Vertex buffer
    float2* h_vb{ static_cast<float2*>(malloc(4 * sizeof(float2))) };
    h_vb[0] = float2(-0.5f, 0.5f);
    h_vb[1] = float2(-0.5f, -0.5f);
    h_vb[2] = float2(0.5f, 0.5f);
    h_vb[3] = float2(0.5f, -0.5f);
    float2* d_vb;
    CC(cudaMalloc(&d_vb, 4 * sizeof(float2)));
    CC(cudaMemcpy(d_vb, h_vb, 4 * sizeof(float2), cudaMemcpyHostToDevice));
    
    //Index buffer (CW winding)
    std::uint32_t* h_ib{ static_cast<std::uint32_t*>(malloc(6 * sizeof(std::uint32_t))) };
    h_ib[0] = 0;
    h_ib[1] = 1;
    h_ib[2] = 2;
    h_ib[3] = 2;
    h_ib[4] = 1;
    h_ib[5] = 3;
    std::uint32_t* d_ib;
    CC(cudaMalloc(&d_ib, 6 * sizeof(std::uint32_t)));
    CC(cudaMemcpy(d_ib, h_ib, 6 * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    
    
    //Main loop
    bool running{ true };
    SDL_Event e;
    while (running)
    {
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT) { running = false; }
        }
        
        //Calculate kernel parameters
        constexpr dim3 blockSize{ 16, 16 }; //256 threads
        constexpr dim3 gridSize{ (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y };
        
        //Render
        Launch_kClearSurface(gridSize, blockSize, &surface, { 160, 230, 255, 255 });
        GlobalSynchronise();
        Launch_kRender(gridSize, blockSize, &surface, width, height, d_vb, d_ib, 6);
        
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