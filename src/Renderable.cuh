#ifndef RENDERABLE_CUH
#define RENDERABLE_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"

class Renderable {
public:
    Vector3f x, n;
    Vector2f u;
    __host__ __device__ Renderable();
    __host__ __device__ ~Renderable();
};

#endif