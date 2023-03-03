#include "MathHelper.cuh"

void eigenvalueDecomposition(const Matrix2x2f& A, Matrix2x2f& Q, Vector2f& l) {
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