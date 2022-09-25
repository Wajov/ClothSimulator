cmake_minimum_required(VERSION 3.20)
project(glad)

include_directories(include)
aux_source_directory(src GLAD_SOURCES)
add_library(glad ${GLAD_SOURCES})
target_include_directories(glad INTERFACE include)
