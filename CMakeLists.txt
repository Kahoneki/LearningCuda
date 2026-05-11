cmake_minimum_required(VERSION 3.18)

project(MyCudaProject
        LANGUAGES CXX CUDA
        VERSION 1.0
)

# ---- CUDA Architecture Settings ----
# Use "native" to auto-detect your GPU's architecture (CMake 3.24+)
# Or hardcode: sm_75, sm_80, sm_86, sm_89, sm_90, etc.
set(CMAKE_CUDA_ARCHITECTURES "native")

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Enable useful warnings ----
add_compile_options(-Wall -Wextra -Wpedantic)

# ---- Executable ----
add_executable(my_cuda_app
        src/main.cu
        src/kernel.cu
)

# ---- Include directories ----
target_include_directories(my_cuda_app
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- Link CUDA libraries (add as needed) ----
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_cuda_app
        PRIVATE
        CUDA::cudart
        # CUDA::cublas    # uncomment if needed
        # CUDA::curand    # uncomment if needed
)