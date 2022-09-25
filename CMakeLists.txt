cmake_minimum_required(VERSION 3.20)
project(ClothSimulator LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(src)
file(GLOB_RECURSE SOURCE "src/*.cpp" "src/*.hpp")
add_executable(ClothSimulator ${SOURCE})

add_subdirectory(test)

# OpenGL
find_package(OpenGL REQUIRED)

# GLAD
add_subdirectory(lib/glad)

# GLFW
find_package(glfw3 REQUIRED)

# jsoncpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# Eigen
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
set_target_properties(ClothSimulator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(
        ClothSimulator
        glad
        glfw
        ${JSONCPP_LIBRARIES}
)