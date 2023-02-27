#ifndef IMPACT_CUH
#define IMPACT_CUH

#include "Vector.cuh"
#include "Node.cuh"

class Impact {
public:
    Node* nodes[4];
    float t, w[4];
    Vector3f n;
    __host__ __device__ Impact();
    __host__ __device__ ~Impact();
    __host__ __device__ bool operator<(const Impact& impact) const;
};

#endif