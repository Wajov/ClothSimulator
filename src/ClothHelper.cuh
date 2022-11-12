#ifndef CLOTH_HELPER_CUH
#define CLOTH_HELPER_CUH

#include <device_launch_parameters.h>

#include "Vertex.cuh"
#include "Handle.cuh"

__global__ static void initializeHandles(int nHandles, const int* handleIndices, Vertex** vertices, Handle** handles) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads) {
        int index = handleIndices[i];
        vertices[index]->preserve = true;
        handles[i] = new Handle(vertices[index], vertices[index]->x);
    }
}

__global__ static void deleteHandles(int nHandles, const Handle* const* handles) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads)
        delete handles[i];
}

#endif