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

float signedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) {
    n = (y1 - y0).normalized().cross((y2 - y0).normalized());
    if (n.norm2() < 1e-6f)
        return INFINITY;
    n.normalize();
    float h = (x - y0).dot(n);
    float b0 = mixed(y1 - x, y2 - x, n);
    float b1 = mixed(y2 - x, y0 - x, n);
    float b2 = mixed(y0 - x, y1 - x, n);
    w[0] = 1.0f;
    w[1] = -b0 / (b0 + b1 + b2);
    w[2] = -b1 / (b0 + b1 + b2);
    w[3] = -b2 / (b0 + b1 + b2);
    return h;
}

float signedEdgeEdgeDistance(const Vector3f& x0, const Vector3f& x1, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float* w) {
    n = (x1 - x0).normalized().cross((y1 - y0).normalized());
    if (n.norm2() < 1e-8f) {
        Vector3f e0 = (x1 - x0).normalized(), e1 = (y1 - y0).normalized();
        float p0min = x0.dot(e0), p0max = x1.dot(e0), p1min = y0.dot(e0), p1max = y1.dot(e0);
        if (p1max < p1min)
            mySwap(p1max, p1min);
        
        float a = max(p0min, p1min), b = min(p0max, p1max), c = 0.5f * (a + b);
        if (a > b)
            return INFINITY;
        
        Vector3f d = y0 - x0 - (y0-x0).dot(e0) * e0;
        n = (-d).normalized();
        w[1] = (c - x0.dot(e0)) / (x1 - x0).norm();
        w[0] = 1.0f - w[1];
        w[3] = -(e0.dot(e1) * c - y0.dot(e1)) / (y1-y0).norm();
        w[2] = -1.0f - w[3];
        return d.norm();
    }
    n = n.normalized();
    float h = (x0 - y0).dot(n);
    float a0 = mixed(y1 - x1, y0 - x1, n);
    float a1 = mixed(y0 - x0, y1 - x0, n);
    float b0 = mixed(x0 - y1, x1 - y1, n);
    float b1 = mixed(x1 - y0, x0 - y0, n);
    w[0] = a0 / (a0 + a1);
    w[1] = a1 / (a0 + a1);
    w[2] = -b0 / (b0 + b1);
    w[3] = -b1 / (b0 + b1);
    return h;
}

float unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1) {
    float t = clamp((x - y0).dot(y1 - y0)/(y1 - y0).dot(y1 - y0), 0.0f, 1.0f);
    Vector3f y = y0 + t * (y1 - y0);
    float d = (x - y).norm();
    n = (x - y).normalized();
    wx = 1.0f;
    wy0 = 1.0f - t;
    wy1 = t;
    return d;
}

float unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) {
    Vector3f nt = (y1 - y0).cross(y2 - y0).normalized();
    float d = abs((x - y0).dot(nt));
    float b0 = mixed(y1 - x, y2 - x, nt);
    float b1 = mixed(y2 - x, y0 - x, nt);
    float b2 = mixed(y0 - x, y1 - x, nt);
    if (b0 >= 0.0f && b1 >= 0.0f && b2 >= 0.0f) {
        n = nt;
        w[0] = 1.0f;
        w[1] = -b0 / (b0 + b1 + b2);
        w[2] = -b1 / (b0 + b1 + b2);
        w[3] = -b2 / (b0 + b1 + b2);
        return d;
    }
    d = INFINITY;
    if (b0 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y1, y2, n, w[0], w[2], w[3]);
        if (dt < d) {
            d = dt;
            w[1] = 0.0f;
        }
    }
    if (b1 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y2, y0, n, w[0], w[3], w[1]);
        if (dt < d) {
            d = dt;
            w[2] = 0.0f;
        }
    }
    if (b2 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y0, y1, n, w[0], w[1], w[2]);
        if (dt < d) {
            d = dt;
            w[3] = 0.0f;
        }
    }
    return d;
}