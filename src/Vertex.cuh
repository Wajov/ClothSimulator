#ifndef VERTEX_CUH
#define VERTEX_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Matrix.cuh"
#include "Bounds.hpp"

class Vertex {
public:
    int index;
    Vector3f x0, x1, x, n, v;
    Vector2f u;
    Matrix2x2f sizing;
    float m, a;
    bool isFree, preserve;
    __host__ __device__ Vertex(const Vector3f& x, bool isFree);
    __host__ __device__ ~Vertex();
    __host__ __device__ Bounds bounds(bool ccd) const;
    __host__ __device__ Vector3f position(float t) const;
};

#endif