#include "kernel.h"
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
    constexpr std::uint32_t n{ 50'000'000 };
    constexpr std::size_t bytes{ n * sizeof(float) };
    
    //Allocate host memory
    float* h_a{ static_cast<float*>(malloc(bytes)) };
    float* h_b{ static_cast<float*>(malloc(bytes)) };
    float* h_c{ static_cast<float*>(malloc(bytes)) };
    
    //Initialise
    for (std::size_t i{ 0 }; i < n; ++i)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    //Allocate device memory
    float* d_a;
    float* d_b;
    float* d_c;
    CC(cudaMalloc(&d_a, bytes));
    CC(cudaMalloc(&d_b, bytes));
    CC(cudaMalloc(&d_c, bytes));
    
    //Copy to device
    CC(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    //Launch kernel
    std::cout << "STARTING" << std::endl;
    launch_vector_add(d_a, d_b, d_c, n);
    std::cout << "FINISHED" << std::endl;
    
    
    //Copy to host
    CC(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    //Verify
    // bool passed{ true };
    // for (std::size_t i{ 0 }; i < n; ++i)
    // {
    //     if (h_c[i] != 3.0f) { passed = false; break; }
    // }
    // std::cout << "Result: " << (passed ? "Passed." : "Failed.") << '\n';
    
    //Cleanup
    CC(cudaFree(d_a));
    CC(cudaFree(d_b));
    CC(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}