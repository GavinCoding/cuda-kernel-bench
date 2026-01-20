cmake_minimum_required(VERSION 3.24)
project(cuda_app LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(SOURCES
    src/main.cu
    src/reduce.cu
    src/matrix_mul_naive.cu
    src/matrix_mul_tiled.cu
    src/cublas_matMul.cu
    src/cuda_utils.cu
)

add_executable(cuda_app_debug ${SOURCES})
add_executable(cuda_app ${SOURCES})


target_include_directories(cuda_app PRIVATE include)

set_target_properties(cuda_app PROPERTIES
    CUDA_ARCHITECTURES 89
)

set_target_properties(cuda_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuda_app PRIVATE CUDA::cublas)

target_compile_options(cuda_app PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od>

    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>
)
