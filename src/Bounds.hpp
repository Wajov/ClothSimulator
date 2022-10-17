#ifndef BOUNDS_HPP
#define BOUNDS_HPP

#include <cfloat>

#include "TypeHelper.hpp"

class Bounds {
private:
    Vector3f pMin, pMax;
    Vector3f minVector(const Vector3f& a, const Vector3f& b) const;
    Vector3f maxVector(const Vector3f& a, const Vector3f& b) const;

public:
    Bounds();
    ~Bounds();
    void operator+=(const Vector3f& v);
    void operator+=(const Bounds& b);
    Vector3f center() const;
    int longestIndex() const;
    Bounds dilate(float thickness) const;
    bool overlap(const Bounds& b) const;
    bool overlap(const Bounds& b, float thickness) const;
};

#endif