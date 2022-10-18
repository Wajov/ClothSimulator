cmake_minimum_required(VERSION 3.20)
project(alglib)

include_directories(src)
aux_source_directory(src ALGLIB_SOURCES)
add_library(alglib ${ALGLIB_SOURCES})
target_include_directories(alglib INTERFACE src)
