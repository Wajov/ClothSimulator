cmake_minimum_required(VERSION 3.20)

if (WIN32)
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        add_compile_definitions(_USE_MATH_DEFINES)
endif()

project(ClothSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(src)
file(GLOB_RECURSE SOURCE "src/*.cpp" "src/*.hpp")
add_executable(ClothSimulator ${SOURCE})

add_subdirectory(test)
add_subdirectory(CudaTest)

# OpenGL
find_package(OpenGL REQUIRED)

# GLAD
add_subdirectory(lib/glad)

# GLFW
find_package(glfw3 REQUIRED)

# jsoncpp
if (WIN32)
        find_package(jsoncpp CONFIG REQUIRED)
elseif()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(JSONCPP jsoncpp)
endif()

# Eigen
find_package(Eigen3 REQUIRED)

# ALGLIB
add_subdirectory(lib/alglib)

# CUDA
find_package(CUDA REQUIRED)

set_target_properties(ClothSimulator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
include_directories(
        ${OPENGL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
)
if (WIN32)
        target_link_libraries(
                ClothSimulator
                glad
                glfw
                JsonCpp::JsonCpp
                alglib
        )
else()
        target_link_libraries(
                ClothSimulator
                glad
                glfw
                jsoncpp
                alglib
        )
endif()
