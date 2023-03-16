#ifndef MATH_HELPER_CUH
#define MATH_HELPER_CUH

#include <vector>

#include "Vector.cuh"
#include "Matrix.cuh"

template<typename T> __host__ __device__ static void mySwap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

template<typename T> __host__ __device__ static T min(T a, T b) {
    return a < b ? a : b;
}

template<typename T> __host__ __device__ static T max(T a, T b) {
    return a > b ? a : b;
}

template<typename T> __host__ __device__ static T abs(T x) {
    return x < static_cast<T>(0) ? -x : x;
}

template<typename T> __host__ __device__ static T clamp(T x , T a, T b) {
    return x < a ? a : (x > b ? b : x);
}

template<typename T> __host__ __device__ static T sqr(T x) {
    return x * x;
}

template<typename T> __host__ __device__ static T min(T a, T b, T c) {
    return min(a, min(b, c));
}

template<typename T> __host__ __device__ static T min(T a, T b, T c, T d) {
    return min(min(a, b), min(c, d));
}

template<typename T> __host__ __device__ static T mixed(const Vector<T, 3>& a, const Vector<T, 3>& b, const Vector<T, 3>& c) {
    return a.dot(b.cross(c));
}

template<typename T> __host__ __device__ static int sign(T x) {
    return x < static_cast<T>(0) ? -1 : 1;
}

__host__ __device__ void eigenvalueDecomposition(const Matrix2x2f& A, Matrix2x2f& Q, Vector2f& l);
__host__ __device__ float signedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w);
__host__ __device__ float signedEdgeEdgeDistance(const Vector3f& x0, const Vector3f& x1, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float* w);
__host__ __device__ float unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1);
__host__ __device__ float unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w);

#endif