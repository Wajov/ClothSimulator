#ifndef MATH_HELPER_CUH
#define MATH_HELPER_CUH

#include <vector>

#include "Vector.cuh"
#include "Matrix.cuh"
#include "optimization/Optimization.cuh"

const int MAX_ITERATIONS = 100;
const float EPSILON = 1e-12f;
const float RHO = 0.9992f;
const float RHO2 = RHO * RHO;

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

static float newtonsMethod(float a, float b, float c, float d, float x0, int dir) {
    if (dir != 0) {
        float y0 = d + x0 * (c + x0 * (b + x0 * a));
        float ddy0 = 2.0f * b + x0 * (6.0f * a);
        x0 += dir * sqrt(abs(2.0f * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        float y = d + x0 * (c + x0 * (b + x0 * a));
        float dy = c + x0 * (2*b + 3.0f * x0 * a);
        if (dy == 0)
            return x0;
        float x1 = x0 - y / dy;
        if (abs(x0 - x1) < 1e-6f)
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
    if (abs(a) > 1e-12 * abs(q))
        x[i++] = q / a;
    if (abs(q) > 1e-12 * abs(c))
        x[i++] = c / q;
    if (i == 2 && x[0] > x[1])
        mySwap(x[0], x[1]);
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

static float clampViolation(float x, int sign) {
    return sign < 0 ? max(x, 0.0f) : (sign > 0 ? min(x, 0.0f) : x);
}

static float value(const Optimization* optimization, const std::vector<float>& lambda, float mu, const std::vector<Vector3f>& x) {
    float ans = optimization->objective(x);
    for (int i = 0; i < optimization->getConstraintSize(); i++) {
        int sign;
        float constraint = optimization->constraint(x, i, sign);
        float coefficient = clampViolation(constraint + lambda[i] / mu, sign);
        if (coefficient != 0.0f)
            ans += 0.5f * mu * sqr(coefficient);
    }
    return ans;
}

static void valueAndGradient(const Optimization* optimization, const std::vector<float>& lambda, float mu, const std::vector<Vector3f>& x, float& value, std::vector<Vector3f>& gradient) {
    value = optimization->objective(x);
    optimization->objectiveGradient(x, gradient);

    for (int i = 0; i < optimization->getConstraintSize(); i++) {
        int sign;
        float constraint = optimization->constraint(x, i, sign);
        float coefficient = clampViolation(constraint + lambda[i] / mu, sign);
        if (coefficient != 0.0f) {
            value += 0.5f * mu * sqr(coefficient);
            optimization->constraintGradient(x, i, mu * coefficient, gradient);
        }
    }
}

static void updateMultiplier(const Optimization* optimization, const std::vector<Vector3f>& x, float mu, std::vector<float>& lambda) {
    for (int i = 0; i < optimization->getConstraintSize(); i++) {
        int sign;
        float constraint = optimization->constraint(x, i, sign);
        lambda[i] = clampViolation(lambda[i] + mu * constraint, sign);
    }
}

static void augmentedLagrangianMethod(Optimization* optimization) {
    int nNodes = optimization->getNodeSize();
    float s = 1e-3f, f, mu = 1e3f, omega = 1.0f;
    std::vector<Vector3f> x(nNodes), gradient(nNodes), t(nNodes);
    std::vector<float> lambda(optimization->getConstraintSize(), 0.0f);
    optimization->initialize(x);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        valueAndGradient(optimization, lambda, mu, x, f, gradient);
       
        float norm2 = 0;
        for (int i = 0; i < nNodes; i++)
            norm2 += gradient[i].norm2();
        s /= 0.7f;
        do {
            s *= 0.7f;
            for (int i = 0; i < nNodes; i++)
                t[i] = x[i] - s * gradient[i];
        } while (value(optimization, lambda, mu, t) >= f - 0.5f * s * norm2 && s >= EPSILON);
        if (s < EPSILON)
            break;

        if (iter == 10)
            omega = 2.0f / (2.0f - RHO2);
        else if (iter > 10)
            omega = 4.0f / (4.0f - RHO2 * omega);
        float coeffient = (1 + omega) * s;
        for (int i = 0; i < nNodes; i++)
            x[i] = x[i] - coeffient * gradient[i];
        
        updateMultiplier(optimization, x, mu, lambda);
    }
    optimization->finalize(x);
}

#endif