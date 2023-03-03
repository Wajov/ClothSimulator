#ifndef DISK_CUH
#define DISK_CUH

#include "MathHelper.cuh"
#include "Vector.cuh"

class Disk {
public:
    Vector2f o;
    float r;
    __host__ __device__ Disk();
    __host__ __device__ Disk(const Vector2f& o, float r);
    __host__ __device__ ~Disk();
    __host__ __device__ bool enclose(const Disk& d) const;
};

#endif