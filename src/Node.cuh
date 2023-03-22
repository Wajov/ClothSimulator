#ifndef NODE_CUH
#define NODE_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Bounds.cuh"

class Node {
public:
    int index, minIndex;
    Vector3f x0, x1, x, n, v;
    float area, mass;
    bool isFree, preserve, removed;
    __host__ __device__ Node();
    __host__ __device__ Node(const Vector3f& x, bool isFree);
    __host__ __device__ ~Node();
    __host__ __device__ Bounds bounds(bool ccd) const;
    __host__ __device__ Vector3f position(float t) const;
};

#endif