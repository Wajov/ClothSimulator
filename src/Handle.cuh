#ifndef HANDLE_CUH
#define HANDLE_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Node.cuh"

class Handle {
public:
    int motionIndex;
    Node* node;
    Vector3f position;
    __host__ __device__ Handle();
    __host__ __device__ ~Handle();
};

#endif