#ifndef QUATERNION_CUH
#define QUATERNION_CUH

#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "Vector.cuh"

class Quaternion {
private:
    float s;
    Vector3f v;

public:
    __host__ __device__ Quaternion();
    __host__ __device__ Quaternion(const Vector3f& axis, float angle);
    __host__ __device__ ~Quaternion();
    __host__ __device__ Vector3f rotate(const Vector3f& x) const;
    __host__ __device__ Quaternion operator+(const Quaternion& q) const;
    __host__ __device__ Quaternion operator-(const Quaternion& q) const;
    __host__ __device__ friend Quaternion operator*(float s, const Quaternion& q);
    __host__ __device__ Quaternion operator*(float s) const;
    __host__ __device__ Quaternion operator/(float s) const;
};

#endif