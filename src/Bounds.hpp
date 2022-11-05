#ifndef BOUNDS_HPP
#define BOUNDS_HPP

#include <algorithm>
#include <cfloat>

#include "Vector.hpp"

class Bounds {
private:
    Vector3f pMin, pMax;
    Vector3f minVector(const Vector3f& a, const Vector3f& b) const;
    Vector3f maxVector(const Vector3f& a, const Vector3f& b) const;

public:
    Bounds();
    Bounds(const Vector3f& pMin, const Vector3f& pMax);
    ~Bounds();
    Bounds operator+(const Bounds& b) const;
    void operator+=(const Vector3f& v);
    void operator+=(const Bounds& b);
    Vector3f center() const;
    int longestIndex() const;
    Bounds dilate(float thickness) const;
    float distance(const Vector3f& x) const;
    bool overlap(const Bounds& b) const;
    bool overlap(const Bounds& b, float thickness) const;
};

#endif