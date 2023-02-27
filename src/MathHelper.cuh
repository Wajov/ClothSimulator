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

template<typename T> __host__ __device__ static T clamp(T x , T a, T b) {
    return x < a ? a : (x > b ? b : x);
}

template<typename T> __host__ __device__ static T sign(T x) {
    return x < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
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

static void eigenvalueDecomposition(const Matrix2x2f& A, Matrix2x2f& Q, Vector2f& l) {
    float a = A(0, 0), b = A(1, 0), d = A(1, 1);
    float amd = a - d;
    float apd = a + d;
    float b2 = b * b;
    float det = sqrt(4.0f * b2 + sqr(amd));
    float l1 = 0.5f * (apd + det);
    float l2 = 0.5f * (apd - det);
    l(0) = l1;
    l(1) = l2;

    float v0, v1, vn;
    if (b != 0.0f) {
        v0 = l1 - d;
        v1 = b;
        vn = sqrt(sqr(v0) + b2);
        Q(0,0) = v0 / vn;
        Q(1,0) = v1 / vn;
        v0 = l2 - d;
        vn = sqrt(sqr(v0) + b2);
        Q(0, 1) = v0 / vn;
        Q(1, 1) = v1 / vn;
    } else if (a >= d) {
        Q(0, 0) = 1.0f;
        Q(1, 0) = 0.0f;
        Q(0, 1) = 0.0f;
        Q(1, 1) = 1.0f;
    } else {
        Q(0, 0) = 0.0f;
        Q(1, 0) = 1.0f;
        Q(0, 1) = 1.0f;
        Q(1, 1) = 0.0f;
    }
}

template<typename T, int n> static Matrix<T, n, n> diagonal(const Vector<T, n>& v) {
    Matrix<T, n, n> ans;
    for (int i = 0; i < n; i++)
        ans(i, i) = v(i);
    return ans;
}

static Matrix2x2f sqrt(const Matrix2x2f& A) {
    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(A, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = l(i) >= 0.0f ? sqrt(l(i)) : -sqrt(-l(i));
    return Q * diagonal(l) * Q.transpose();
}

static Matrix2x2f max(const Matrix2x2f& A, float v) {
    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(A, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = max(l(i), v);
    return Q * diagonal(l) * Q.transpose();
}

#endif