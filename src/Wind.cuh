#ifndef WIND_CUH
#define WIND_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"

class Wind {
private:
    float density, drag;
    Vector3f velocity;

public:
    __host__ __device__ Wind();
    __host__ __device__ ~Wind();
    __host__ __device__ float getDensity() const;
    __host__ __device__ float getDrag() const;
    __host__ __device__ Vector3f getVelocity() const;
};

#endif