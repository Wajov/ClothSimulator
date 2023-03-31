#ifndef CLOTH_HELPER_CUH
#define CLOTH_HELPER_CUH

#include <device_launch_parameters.h>

#include "Vertex.cuh"
#include "Handle.cuh"
#include "Transformation.cuh"

__global__ void initializeHandles(int nHandles, const int* handleIndices, int motionIndex, const Transformation* transformations, Node** nodes, Handle* handles);
__global__ void collectHandleIndices(int nHandles, const Handle* handles, int* handleIndices);

#endif