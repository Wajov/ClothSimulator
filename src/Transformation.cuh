#ifndef TRANSFORMATION_CUH
#define TRANSFORMATION_CUH

#include <json/json.h>
#include <cuda_runtime.h>

#include "JsonHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Quaternion.cuh"

class Transformation {
private:
    float scaling;
    Vector3f translation;
    Quaternion rotation;

public:
    __host__ __device__ Transformation();
    __host__ __device__ Transformation(const Json::Value& json);
    __host__ __device__ Transformation(const Vector3f& translation, const Quaternion& rotation);
    __host__ __device__ ~Transformation();
    __host__ __device__ Transformation operator+(const Transformation& t) const;
    __host__ __device__ Transformation operator-(const Transformation& t) const;
    __host__ __device__ friend Transformation operator*(float s, const Transformation& t);
    __host__ __device__ Transformation operator*(float s) const;
    __host__ __device__ Transformation operator*(const Transformation& t) const;
    __host__ __device__ Transformation operator/(float s) const;
    __host__ __device__ Vector2f applyToUV(const Vector2f& u) const;
    __host__ __device__ Vector3f applyToPoint(const Vector3f& p) const;
    __host__ __device__ Vector3f applyToVector(const Vector3f& v) const;
};

#endif