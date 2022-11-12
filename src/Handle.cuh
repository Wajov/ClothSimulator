#ifndef HANDLE_CUH
#define HANDLE_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Vertex.cuh"

class Handle {
private:
    Vertex* vertex;
    Vector3f position;

public:
    __host__ __device__ Handle(const Vertex* vertex, const Vector3f& position);
    __host__ __device__ ~Handle();
    __host__ __device__ Vertex* getVertex() const;
    __host__ __device__ Vector3f getPosition() const;
};

#endif