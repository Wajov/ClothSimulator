#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cmath>

#include <cuda_runtime.h>

#include "Matrix.cuh"

template<typename T, int n, int p> class Matrix;

template<typename T, int n> class Vector {
public:
    T data[n];
    
    __host__ __device__ Vector() {
        for (int i = 0; i < n; i++)
            data[i] = static_cast<T>(0);
    };

    __host__ __device__ Vector(T x, T y) {
        static_assert(n == 2);
        data[0] = x;
        data[1] = y;
    };

    __host__ __device__ Vector(T x, T y, T z) {
        static_assert(n == 3);
        data[0] = x;
        data[1] = y;
        data[2] = z;
    };

    __host__ __device__ Vector(T x, T y, T z, T w) {
        static_assert(n == 4);
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    };

    template<int m> __host__ __device__ Vector(const Vector<T, m>& v0, const Vector<T, m>& v1, const Vector<T, m>& v2) {
        static_assert(n == 3 * m);
        for (int i = 0; i < m; i++) {
            data[i] = v0.data[i];
            data[m + i] = v1.data[i];
            data[2 * m + i] = v2.data[i];
        }
    };

    template<int m> __host__ __device__ Vector(const Vector<T, m>& v0, const Vector<T, m>& v1, const Vector<T, m>& v2, const Vector<T, m>& v3) {
        static_assert(n == 4 * m);
        for (int i = 0; i < m; i++) {
            data[i] = v0.data[i];
            data[m + i] = v1.data[i];
            data[2 * m + i] = v2.data[i];
            data[3 * m + i] = v3.data[i];
        }
    };

    __host__ __device__ ~Vector() {};

    __host__ __device__ const T& operator()(int i) const {
        return data[i];
    };

    __host__ __device__ T& operator()(int i) {
        return data[i];
    };
    
    __host__ __device__ Vector<T, n> operator+(const Vector<T, n>& v) const {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans.data[i] = data[i] + v.data[i];
        return ans;
    };

    __host__ __device__ void operator+=(const Vector<T, n>& v) {
        for (int i = 0; i < n; i++)
            data[i] += v.data[i];
    };

    __host__ __device__ Vector<T, n> operator-() const {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans.data[i] = -data[i];
        return ans;
    };

    __host__ __device__ Vector<T, n> operator-(const Vector<T, n>& v) const {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans.data[i] = data[i] - v.data[i];
        return ans;
    };

    __host__ __device__ void operator-=(const Vector<T, n>& v) {
        for (int i = 0; i < n; i++)
            data[i] -= v.data[i];
    };

    __host__ __device__ friend Vector<T, n> operator*(T s, const Vector<T, n>& v) {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans.data[i] = s * v.data[i];
        return ans;
    };

    __host__ __device__ Vector<T, n> operator*(T s) const {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans.data[i] = data[i] * s;
        return ans;
    };

    __host__ __device__ void operator*=(T s) {
        for (int i = 0; i < n; i++)
            data[i] *= s;
    };

    __host__ __device__ Vector<T, n> operator/(T s) const {
        T invS = static_cast<T>(1) / s;
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans.data[i] = data[i] * invS;
        return ans;
    };

    __host__ __device__ void operator/=(T s) {
        T invS = static_cast<T>(1) / s;
        for (int i = 0; i < n; i++)
            data[i] *= invS;
    };

    __host__ __device__ T norm() const {
        return sqrt(norm2());
    };

    __host__ __device__ T norm2() const {
        T ans = static_cast<T>(0);
        for (int i = 0; i < n; i++)
            ans += data[i] * data[i];
        return ans;
    };

    __host__ __device__ void normalize() {
        T l = norm();
        *this /= l;
    };

    __host__ __device__ Vector<T, n> normalized() const {
        T l = norm();
        return *this / l;
    };

    __host__ __device__ T dot(const Vector<T, n>& v) const {
        T ans = static_cast<T>(0);
        for (int i = 0; i < n; i++)
            ans += data[i] * v.data[i];
        return ans;
    };

    __host__ __device__ T cross(const Vector<T, 2>& v) const {
        static_assert(n == 2);
        return data[0] * v.data[1] - data[1] * v.data[0];
    };

    __host__ __device__ Vector<T, n> cross(const Vector<T, 3>& v) const {
        static_assert(n == 3);
        return Vector<T, n>(data[1] * v.data[2] - data[2] * v.data[1], data[2] * v.data[0] - data[0] * v.data[2], data[0] * v.data[1] - data[1] * v.data[0]);
    };

    template<int p> __host__ __device__ Matrix<T, n, p> outer(const Vector<T, p>& v) {
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans(i, j) = data[i] * v.data[j];
        return ans;
    };
};

typedef Vector<int, 3> Vector3i;
typedef Vector<int, 4> Vector4i;
typedef Vector<float, 2> Vector2f;
typedef Vector<float, 3> Vector3f;
typedef Vector<float, 4> Vector4f;
typedef Vector<float, 9> Vector9f;
typedef Vector<float, 12> Vector12f;

#endif