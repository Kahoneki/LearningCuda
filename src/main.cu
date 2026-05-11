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
    SDL_Window* window{ SDL_CreateWindow("CUDA Software Rasteriser", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0) };
    SDL_Surface* surface{ SDL_GetWindowSurface(window) };
    
    //Allocate pixel buffer
    uchar4* d_pixels;
    CC(cudaMalloc(&d_pixels, width * height * sizeof(uchar4)));
    
    bool running{ true };
    SDL_Event e;
    while (running)
    {
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT) { running = false; }
        }
        
        //Launch the kernel
        constexpr dim3 blockSize{ 16, 16 }; //256 threads
        constexpr dim3 gridSize{ (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y };
        Launch_RenderKernel(gridSize, blockSize, d_pixels, width, height);
        
        //Copy pixels from GPU to CPU
        //Lock the surface so SDL doesn't overwrite it while we write
        SDL_LockSurface(surface);
        CC(cudaMemcpy(surface->pixels, d_pixels, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
        SDL_UnlockSurface(surface);
        
        //Push CPU pixels to the monitor
        SDL_UpdateWindowSurface(window);
    }
    
    //Cleanup
    CC(cudaFree(d_pixels));
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}