#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Vector.hpp"

template<typename T, int n> class Vector;

template<typename T, int n, int p> class Matrix {
public:
    T data[n][p];

    Matrix() {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i][j] = static_cast<T>(0);
    };

    Matrix(T s) {
        static_assert(n == p);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i][j] = static_cast<T>(0);
        for (int i = 0; i < n; i++)
            data[i][i] = s;
    };

    Matrix(const Vector<T, n>& v0, const Vector<T, n>& v1) {
        static_assert(p == 2);
        for (int i = 0; i < n; i++) {
            data[i][0] = v0(i);
            data[i][1] = v1(i);
        }
    };

    Matrix(const Vector<T, n>& v0, const Vector<T, n>& v1, const Vector<T, n>& v2) {
        static_assert(p == 3);
        for (int i = 0; i < n; i++) {
            data[i][0] = v0(i);
            data[i][1] = v1(i);
            data[i][2] = v2(i);
        }
    };

    template<int q> Matrix(const Matrix<T, n, q>& m0, const Matrix<T, n, q>& m1, const Matrix<T, n, q>& m2) {
        static_assert(p == 3 * q);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < q; j++) {
                data[i][j] = m0.data[i][j];
                data[i][q + j] = m1.data[i][j];
                data[i][2 * q + j] = m2.data[i][j];
            }
    }

    ~Matrix() {};

    const T& operator()(int i, int j) const {
        return data[i][j];
    };

    T& operator()(int i, int j) {
        return data[i][j];
    };

    Matrix<T, n, p> operator+(const Matrix<T, n, p>& m) const {
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i][j] = data[i][j] + m.data[i][j];
        return ans;
    };

    void operator+=(const Matrix<T, n, p>& m) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i][j] += m.data[i][j];
    };

    Matrix<T, n, p> operator-() const {
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i][j] = -data[i][j];
        return ans;
    };

    Matrix<T, n, p> operator-(const Matrix<T, n, p>& m) const {
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i][j] = data[i][j] - m.data[i][j];
        return ans;
    };

    void operator-=(const Matrix<T, n, p>& m) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i][j] -= m.data[i][j];
    };

    friend Matrix<T, n, p> operator*(T s, const Matrix<T, n, p>& m) {
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i][j] = s * m.data[i][j];
        return ans;
    };

    Matrix<T, n, p> operator*(T s) const {
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i][j] = data[i][j] * s;
        return ans;
    };

    void operator*=(T s) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i][j] *= s;
    };

    Vector<T, n> operator*(const Vector<T, p>& v) const {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i] += data[i][j] * v(j);
        return ans;
    };

    template<int q> Matrix<T, n, q> operator*(const Matrix<T, p, q>& m) const {
        Matrix<T, n, q> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < q; j++)
                for (int k = 0; k < p; k++)
                    ans.data[i][j] += data[i][k] * m.data[k][j];
        return ans;
    };

    Matrix<T, n, p> operator/(T s) const {
        T invS = static_cast<T>(1) / s;
        Matrix<T, n, p> ans;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                ans.data[i][j] = data[i][j] * invS;
        return ans;
    };

    void operator/=(T s) {
        T invS = static_cast<T>(1) / s;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i][j] *= invS;
    };

    Matrix<T, p, n> transpose() const {
        Matrix<T, p, n> ans;
        for (int i = 0; i < p; i++)
            for (int j = 0; j < n; j++)
                ans.data[i][j] = data[j][i];
        return ans;
    };

    T trace() const {
        static_assert(n == p);
        T ans = static_cast<T>(0);
        for (int i = 0; i < n; i++)
            ans += data[i][i];
        return ans;
    }

    Matrix<T, n, p> inverse() const {
        static_assert(n == 2 && p == 2);
        Matrix<T, n, p> ans;
        T a = data[0][0];
        T b = data[0][1];
        T c = data[1][0];
        T d = data[1][1];
        T invDet = static_cast<T>(1) / (a * d - b * c);
        ans.data[0][0] = d * invDet;
        ans.data[0][1] = -b * invDet;
        ans.data[1][0] = -c * invDet;
        ans.data[1][1] = a * invDet;
        return ans;
    }

    Vector<T, p> row(int r) const {
        Vector<T, p> ans;
        for (int i = 0; i < p; i++)
            ans(i) = data[r][i];
        return ans;
    };

    Vector<T, n> col(int c) const {
        Vector<T, n> ans;
        for (int i = 0; i < n; i++)
            ans(i) = data[i][c];
        return ans;
    };
};

typedef Matrix<float, 2, 2> Matrix2x2f;
typedef Matrix<float, 3, 3> Matrix3x3f;
typedef Matrix<float, 4, 4> Matrix4x4f;
typedef Matrix<float, 2, 3> Matrix2x3f;
typedef Matrix<float, 3, 2> Matrix3x2f;
typedef Matrix<float, 3, 9> Matrix3x9f;
typedef Matrix<float, 9, 9> Matrix9x9f;
typedef Matrix<float, 12, 12> Matrix12x12f;

#endif