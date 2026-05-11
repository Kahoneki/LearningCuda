cmake_minimum_required(VERSION 3.18)

project(MyCudaProject
        LANGUAGES CXX CUDA
        VERSION 1.0
)

# ---- CUDA Architecture Settings ----
set(CMAKE_CUDA_ARCHITECTURES "native")

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Enable useful warnings ----
add_compile_options(-Wall -Wextra -Wpedantic)

# ---- FetchContent for SDL2 ----
include(FetchContent)

set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-2.30.8   # pin a specific release
)

FetchContent_MakeAvailable(SDL2)

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

# ---- Link libraries ----
find_package(CUDAToolkit REQUIRED)

target_link_libraries(my_cuda_app
        PRIVATE
        CUDA::cudart
        SDL2::SDL2          # SDL2 (static or shared depending on BUILD_SHARED_LIBS)
        # SDL2::SDL2main    # uncomment on Windows if you use SDL_main
        # CUDA::cublas
        # CUDA::curand
)

# ---- SDL2 compile definitions (optional but recommended) ----
target_compile_definitions(my_cuda_app
        PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)