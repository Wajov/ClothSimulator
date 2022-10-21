#include "Disk.hpp"

Disk::Disk() :
    o(0.0f, 0.0f),
    r(0.0f) {}

Disk::Disk(const Vector2f& o, float r) :
    o(o),
    r(r) {}

Disk::~Disk() {}

bool Disk::enclose(const Disk& d) const {
    return r >= d.r && sqr(r - d.r) >= (o - d.o).squaredNorm();    
}

Disk Disk::circumscribedDisk(const Disk& d0, const Disk& d1) {
    float d = (d0.o - d1.o).norm();
    float r = 0.5f * (d0.r + d + d1.r);
    float t = (r - d0.r) / d;
    return Disk(d0.o + t * (d1.o - d0.o), r);
}

Disk Disk::circumscribedDisk(const Disk& d0, const Disk& d1, const Disk& d2) {
    float x0 = d0.o(0), y0 = d0.o(1), r0 = d0.r;
    float x1 = d1.o(0), y1 = d1.o(1), r1 = d1.r;
    float x2 = d2.o(0), y2 = d2.o(1), r2 = d2.r;

    float v11 = 2.0f * x1 - 2.0f * x0;
    float v12 = 2.0f * y1 - 2.0f * y0;
    float v13 = sqr(x0) - sqr(x1) + sqr(y0) - sqr(y1) - sqr(r0) + sqr(r1);
    float v14 = 2.0f * r1 - 2.0f * r0;
    float v21 = 2.0f * x2 - 2.0f * x1;
    float v22 = 2.0f * y2 - 2.0f * y1;
    float v23 = sqr(x1) - sqr(x2) + sqr(y1) - sqr(y2) - sqr(r1) + sqr(r2);
    float v24 = 2.0f * r2 - 2.0f * r1;
    float w12 = v12 / v11;
    float w13 = v13 / v11;
    float w14 = v14 / v11;
    float w22 = v22 / v21 - w12;
    float w23 = v23 / v21 - w13;
    float w24 = v24 / v21 - w14;
    float P = -w23 / w22;
    float Q = w24 / w22;
    float M = - w12 * P - w13;
    float N = w14 - w12 * Q;
    float a = sqr(N) + sqr(Q) - 1.0f;
    float b = 2.0f * M * N - 2.0f * N * x0 + 2.0f * P * Q - 2.0f * Q * y0 + 2.0f * r0;
    float c = sqr(x0) + sqr(M) - 2.0f * M * x0 + sqr(P) + sqr(y0) - 2.0f * P * y0 - sqr(r0);
    float D = sqr(b) - 4.0f * a * c;
    float rs = (-b - std::sqrt(D)) / (2.0f * a);
    float xs = M + N * rs;
    float ys = P + Q * rs;

    return Disk(Vector2f(xs , ys), rs);
}
