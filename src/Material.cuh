#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <cmath>
#include <fstream>
#include <iostream>

#include <json/json.h>
#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "JsonHelper.hpp"
#include "Vector.cuh"
#include "Matrix.cuh"

const int N = 30;

class Material {
private:
    float density, thicken;
    Vector4f stretchingSamples[N][N][N];
    float bendingSamples[3][5];
    Vector4f calculateStretchingSample(const Matrix2x2f& G, const Vector4f (&data)[2][5]) const;

public:
    Material(const Json::Value& json);
    __host__ __device__ ~Material();
    __host__ __device__ float getDensity() const;
    __host__ __device__ float getThicken() const;
    __host__ __device__ Vector4f stretchingStiffness(const Matrix2x2f& G) const;
    __host__ __device__ float bendingStiffness(float length, float angle, float area, const Vector2f& d) const;
};

#endif