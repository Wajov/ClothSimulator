#ifndef NEAR_POINT_CUH
#define NEAR_POINT_CUH

#include "Vector.cuh"

class NearPoint {
public:
    float d;
    Vector3f x;
    __host__ __device__ NearPoint(float d, const Vector3f& x);
    __host__ __device__ ~NearPoint();
};

#endif