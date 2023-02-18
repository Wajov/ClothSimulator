#ifndef HANDLE_CUH
#define HANDLE_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Node.cuh"

class Handle {
private:
    Node* node;
    Vector3f position;

public:
    __host__ __device__ Handle(const Node* node, const Vector3f& position);
    __host__ __device__ ~Handle();
    __host__ __device__ Node* getNode() const;
    __host__ __device__ Vector3f getPosition() const;
};

#endif