#include "Bounds.cuh"

Bounds::Bounds() :
    pMin(FLT_MAX, FLT_MAX, FLT_MAX),
    pMax(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}

Bounds::Bounds(const Vector3f& pMin, const Vector3f& pMax) :
    pMin(pMin),
    pMax(pMax) {}

Bounds::~Bounds() {}

Vector3f Bounds::minVector(const Vector3f& a, const Vector3f& b) const {
    Vector3f ans;
    for (int i = 0; i < 3; i++)
        ans(i) = min(a(i), b(i));

    return ans;
}

Vector3f Bounds::maxVector(const Vector3f& a, const Vector3f& b) const {
    Vector3f ans;
    for (int i = 0; i < 3; i++)
        ans(i) = max(a(i), b(i));

    return ans;
}

Bounds Bounds::operator+(const Bounds& b) const {
    return Bounds(minVector(pMin, b.pMin), maxVector(pMax, b.pMax));
}

void Bounds::operator+=(const Vector3f& v) {
    pMin = minVector(pMin, v);
    pMax = maxVector(pMax, v);
}

void Bounds::operator+=(const Bounds& b) {
    pMin = minVector(pMin, b.pMin);
    pMax = maxVector(pMax, b.pMax);
}

Vector3f Bounds::center() const {
    return 0.5f * (pMin + pMax);
}

int Bounds::majorAxis() const {
    Vector3f d = pMax - pMin;
    if (d(0) >= d(1) && d(0) >= d(2))
        return 0;
    else if (d(1) >= d(0) && d(1) >= d(2))
        return 1;
    else
        return 2;
}

Bounds Bounds::dilate(float thickness) const {
    Bounds ans = *this;
    for (int i = 0; i < 3; i++) {
        ans.pMin(i) -= thickness;
        ans.pMax(i) += thickness;
    }

    return ans;
}

float Bounds::distance(const Vector3f& x) const {
    Vector3f p;
    for (int i = 0; i < 3; i++)
        p(i) = clamp(x(i), pMin(i), pMax(i));
    return (x - p).norm();
}

bool Bounds::overlap(const Bounds& b) const {
    for (int i = 0; i < 3; i++) {
        if (pMin(i) > b.pMax(i))
            return false;
        if (pMax(i) < b.pMin(i))
            return false;
    }

    return true;
}

bool Bounds::overlap(const Bounds& b, float thickness) const {
    return overlap(b.dilate(thickness));
}
