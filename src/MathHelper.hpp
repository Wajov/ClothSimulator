#ifndef MATH_HELPER_HPP
#define MATH_HELPER_HPP

#include "TypeHelper.hpp"

static float sign(float x) {
    return x < 0.0f ? -1.0f : 1.0f;
}

static float min(float a, float b, float c) {
    return std::min(a, std::min(b, c));
}

static float min(float a, float b, float c, float d) {
    return std::min(std::min(a, b), std::min(c, d));
}

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

static float mixed(const Vector3f& a, const Vector3f& b, const Vector3f& c) {
    return a.dot(b.cross(c));
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

static float newtonsMethod(float a, float b, float c, float d, float x0, int dir) {
    if (dir != 0) {
        float y0 = d + x0 * (c + x0 * (b + x0 * a));
        float ddy0 = 2.0f * b + x0 * (6.0f * a);
        x0 += dir * std::sqrt(std::abs(2.0f * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        double y = d + x0 * (c + x0 * (b + x0 * a));
        double dy = c + x0 * (2*b + 3.0f * x0 * a);
        if (dy == 0)
            return x0;
        double x1 = x0 - y / dy;
        if (std::abs(x0 - x1) < 1e-6f)
            return x0;
        x0 = x1;
    }
    return x0;
}

static int solveQuadratic(float a, float b, float c, float x[2]) {
    float d = b * b - 4.0f * a * c;
    if (d < 0.0f) {
        x[0] = -b / (2.0f * a);
        return 0;
    }
    float q = -(b + sign(b) * sqrt(d)) * 0.5f;
    int i = 0;
    if (std::abs(a) > 1e-12 * std::abs(q))
        x[i++] = q / a;
    if (std::abs(q) > 1e-12 * std::abs(c))
        x[i++] = c / q;
    if (i == 2 && x[0] > x[1])
        std::swap(x[0], x[1]);
    return i;
}

static int solveCubic(float a, float b, float c, float d, float x[]) {
    float xc[2];
    int n = solveQuadratic(3.0f * a, 2.0f * b, c, xc);
    if (n == 0) {
        x[0] = newtonsMethod(a, b, c, d, xc[0], 0);
        return 1;
    } else if (n == 1)
        return solveQuadratic(b, c, d, x);
    else {
        float yc[2] = {d + xc[0] * (c + xc[0] * (b + xc[0] * a)), d + xc[1] * (c + xc[1] * (b + xc[1] * a))};
        int i = 0;
        if (yc[0] * a >= 0.0f)
            x[i++] = newtonsMethod(a, b, c, d, xc[0], -1);
        if (yc[0] * yc[1] <= 0.0f) {
            int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
            x[i++] = newtonsMethod(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
        }
        if (yc[1] * a <= 0.0f)
            x[i++] = newtonsMethod(a, b, c, d, xc[1], 1);
        return i;
    }
}

#endif