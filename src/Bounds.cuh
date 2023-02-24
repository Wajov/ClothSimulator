#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include <cfloat>

#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "Vector.cuh"

class Bounds {
private:
    __host__ __device__ Vector3f minVector(const Vector3f& a, const Vector3f& b) const;
    __host__ __device__ Vector3f maxVector(const Vector3f& a, const Vector3f& b) const;

public:
    Vector3f pMin, pMax;
    __host__ __device__ Bounds();
    __host__ __device__ Bounds(const Vector3f& pMin, const Vector3f& pMax);
    __host__ __device__ ~Bounds();
    __host__ __device__ Bounds operator+(const Bounds& b) const;
    __host__ __device__ void operator+=(const Vector3f& v);
    __host__ __device__ void operator+=(const Bounds& b);
    __host__ __device__ Vector3f center() const;
    __host__ __device__ int majorAxis() const;
    __host__ __device__ Bounds dilate(float thickness) const;
    __host__ __device__ float distance(const Vector3f& x) const;
    __host__ __device__ bool overlap(const Bounds& b) const;
    __host__ __device__ bool overlap(const Bounds& b, float thickness) const;
};

#endif