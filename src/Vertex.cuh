#ifndef VERTEX_CUH
#define VERTEX_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Matrix.cuh"
#include "Node.cuh"

class Vertex {
public:
    int index;
    Vector2f u;
    float area;
    Node* node;
    Matrix2x2f sizing;
    __host__ __device__ Vertex(const Vector2f& u);
    __host__ __device__ ~Vertex();
};

#endif