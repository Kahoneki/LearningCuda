cmake_minimum_required(VERSION 3.18)

project(MyCudaProject
        LANGUAGES CXX CUDA
        VERSION 1.0
)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

include(FetchContent)

set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-2.30.8
)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
)

FetchContent_MakeAvailable(SDL2 glm stb)  # ← assimp removed

add_executable(my_cuda_app
        src/main.cu
        src/kernel.cu
        src/Camera.cpp
        ext/stb_image.cpp
)

target_include_directories(my_cuda_app
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${stb_SOURCE_DIR}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(my_cuda_app
        PRIVATE
        CUDA::cudart
        SDL2::SDL2
        glm::glm
        # assimp::assimp  ← removed
)

target_compile_definitions(my_cuda_app
        PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
        GLM_COMPILER=GLM_COMPILER_CUDA
)

add_custom_command(
        TARGET my_cuda_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:my_cuda_app>/resources
        COMMENT "Copying resources to build folder"
)