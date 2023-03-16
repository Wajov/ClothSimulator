#ifndef OBSTACLE_HELPER_CUH
#define OBSTACLE_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Vector.cuh"
#include "Node.cuh"
#include "Transformation.cuh"

__global__ void setBase(int nNodes, const Node* const* nodes, Vector3f* base);
__global__ void transformGpu(int nNodes, const Vector3f* base, const Transformation transformation, Node** nodes);
__global__ void stepGpu(int nNodes, float invDt, const Vector3f* base, const Transformation transformation, Node** nodes);

#endif