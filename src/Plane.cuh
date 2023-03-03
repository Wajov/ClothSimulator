#ifndef PLANE_CUH
#define PLANE_CUH

#include "Vector.cuh"

class Plane {
public:
    Vector3f p, n;
    __host__ __device__ Plane(const Vector3f& p, const Vector3f& n);
    __host__ __device__ ~Plane();
};

#endif