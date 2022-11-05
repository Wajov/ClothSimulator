cmake_minimum_required(VERSION 3.17)
project(CudaTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)

set_target_properties(ClothSimulator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CudaTest main.cu)

include_directories(
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
    CudaTest
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
)