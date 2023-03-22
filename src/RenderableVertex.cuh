#ifndef RENDERABLE_VERTEX_CUH
#define RENDERABLE_VERTEX_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"

class RenderableVertex {
public:
    Vector3f x, n;
    Vector2f u;
    __host__ __device__ RenderableVertex();
    __host__ __device__ ~RenderableVertex();
};

#endif