#ifndef MATH_HELPER_HPP
#define MATH_HELPER_HPP

#include "TypeHelper.hpp"

static Vector9f concatenateToVector(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) {
    Vector9f ans;
    ans.block<3, 1>(0, 0) = v0;
    ans.block<3, 1>(3, 0) = v1;
    ans.block<3, 1>(6, 0) = v2;
    return ans;
}

static Vector12f concatenateToVector(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& v3) {
    Vector12f ans;
    ans.block<3, 1>(0, 0) = v0;
    ans.block<3, 1>(3, 0) = v1;
    ans.block<3, 1>(6, 0) = v2;
    ans.block<3, 1>(9, 0) = v3;
    return ans;
}

static Matrix3x3f concatenateToMatrix(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) {
    Matrix3x3f ans;
    ans.col(0) = v0;
    ans.col(1) = v1;
    ans.col(2) = v2;
    return ans;
}

static MatrixXxXf kronecker(const MatrixXxXf& A, const MatrixXxXf& B) {
    int n = A.rows(), m = A.cols(), p = B.rows(), q = B.cols();
    MatrixXxXf ans;
    ans.resize(n * p, m * q);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < p; k++)
                for (int h = 0; h < q; h++)
                    ans(i * p + k, j * q + h) = A(i, j) * B(k, h);
    return ans;
}

#endif