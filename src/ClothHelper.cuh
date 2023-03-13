#ifndef CLOTH_HELPER_CUH
#define CLOTH_HELPER_CUH

#include <device_launch_parameters.h>

#include "Vertex.cuh"
#include "Handle.cuh"

__global__ void initializeHandles(int nHandles, const int* handleIndices, Node** nodes, Handle* handles);
__global__ void collectHandleIndices(int nHandles, const Handle* handles, int* handleIndices);

#endif