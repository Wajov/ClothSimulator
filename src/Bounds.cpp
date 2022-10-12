#include "Bounds.hpp"

Bounds::Bounds() :
    pMin(FLT_MAX, FLT_MAX, FLT_MAX),
    pMax(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}

Bounds::~Bounds() {}

Vector3f Bounds::minVector(const Vector3f& a, const Vector3f& b) {
    Vector3f ans;
    for (int i = 0; i < 3; i++)
        ans(i) = std::min(a(i), b(i));
    
    return ans;
}

Vector3f Bounds::maxVector(const Vector3f& a, const Vector3f& b) {
    Vector3f ans;
    for (int i = 0; i < 3; i++)
        ans(i) = std::max(a(i), b(i));
    
    return ans;
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

int Bounds::longestIndex() const {
    Vector3f d = pMax - pMin;
    if (d(0) > d(1) && d(0) > d(2))
        return 0;
    else if (d(1) > d(0) && d(1) > d(2))
        return 1;
    else
        return 2;
}

bool Bounds::overlap(const Bounds& b, float thickness) const {
    
}
