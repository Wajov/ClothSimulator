#ifndef MATH_HELPER_HPP
#define MATH_HELPER_HPP

#include <vector>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "optimization/Optimization.hpp"

const int MAX_TOTAL_ITERATIONS = 100;
const int MAX_SUB_ITERATIONS = 10;
const double EPSILON_G = 1e-6;
const double EPSILON_F = 1e-12;
const double EPSILON_X = 1e-6;

static Optimization* optimization;
static std::vector<double> lambda;
static double mu;

template<typename T> static T sign(T x) {
    return x < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
}

template<typename T> static T sqr(T x) {
    return x * x;
}

template<typename T> static T min(T a, T b, T c) {
    return std::min(a, std::min(b, c));
}

template<typename T> static T min(T a, T b, T c, T d) {
    return std::min(std::min(a, b), std::min(c, d));
}

template<typename T> static T mixed(const Vector<T, 3>& a, const Vector<T, 3>& b, const Vector<T, 3>& c) {
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
        vn = std::sqrt(sqr(v0) + b2);
        Q(0,0) = v0 / vn;
        Q(1,0) = v1 / vn;
        v0 = l2 - d;
        vn = std::sqrt(sqr(v0) + b2);
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
        l(i) = l(i) >= 0.0f ? std::sqrt(l(i)) : -std::sqrt(-l(i));
    return Q * diagonal(l) * Q.transpose();
}

static Matrix2x2f max(const Matrix2x2f& A, float v) {
    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(A, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = std::max(l(i), v);
    return Q * diagonal(l) * Q.transpose();
}

static float newtonsMethod(float a, float b, float c, float d, float x0, int dir) {
    if (dir != 0) {
        float y0 = d + x0 * (c + x0 * (b + x0 * a));
        float ddy0 = 2.0f * b + x0 * (6.0f * a);
        x0 += dir * std::sqrt(std::abs(2.0f * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        float y = d + x0 * (c + x0 * (b + x0 * a));
        float dy = c + x0 * (2*b + 3.0f * x0 * a);
        if (dy == 0)
            return x0;
        float x1 = x0 - y / dy;
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

static double clampViolation (double x, int sign) {
    return sign < 0 ? std::max(x, 0.0) : (sign > 0 ? std::min(x, 0.0) : x);
}

static void valueAndGradient(const alglib::real_1d_array& x, double& value, alglib::real_1d_array& gradient, void* p) {
    optimization->precompute(x.getcontent());
    value = optimization->objective(x.getcontent());
    optimization->objectiveGradient(x.getcontent(), gradient.getcontent());

    for (int i = 0; i < optimization->getConstraintSize(); i++) {
        int sign;
        double constraint = optimization->constraint(x.getcontent(), i, sign);
        double coefficient = clampViolation(constraint + lambda[i] / mu, sign);
        if (coefficient != 0.0) {
            value += 0.5 * mu * sqr(coefficient);
            optimization->constraintGradient(x.getcontent(), i, mu * coefficient, gradient.getcontent());
        }
    }
}

static void updateMultiplier(const alglib::real_1d_array& x) {
    optimization->precompute(x.getcontent());
    for (int i = 0; i < optimization->getConstraintSize(); i++) {
        int sign;
        double constraint = optimization->constraint(x.getcontent(), i, sign);
        lambda[i] = clampViolation(lambda[i] + mu * constraint, sign);
    }
}

static void augmentedLagrangianMethod(Optimization* optimization) {
    ::optimization = optimization;
    lambda.assign(::optimization->getConstraintSize(), 0.0);
    mu = 1e3;
    alglib::real_1d_array x;
    x.setlength(::optimization->getVariableSize());
    ::optimization->initialize(x.getcontent());
    alglib::mincgstate state;
    alglib::mincgreport report;
    alglib::mincgcreate(x, state);
    
    int iter = 0;
    while (iter < MAX_TOTAL_ITERATIONS) {
        int maxIterations = std::min(MAX_SUB_ITERATIONS, MAX_TOTAL_ITERATIONS - iter);
        alglib::mincgsetcond(state, EPSILON_G, EPSILON_F, EPSILON_X, maxIterations);
        if (iter > 0)
            alglib::mincgrestartfrom(state, x);
        alglib::mincgsuggeststep(state, 1e-3 * ::optimization->getVariableSize());
        alglib::mincgoptimize(state, valueAndGradient);
        alglib::mincgresults(state, x, report);
        updateMultiplier(x);
        if (report.iterationscount == 0)
            break;
        iter += report.iterationscount;
    }
    ::optimization->finalize(x.getcontent());
}

#endif