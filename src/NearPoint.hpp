#ifndef NEAR_POINT_HPP
#define NEAR_POINT_HPP

#include "Vector.cuh"

class NearPoint {
public:
    float d;
    Vector3f x;
    NearPoint(float d, const Vector3f& x);
    ~NearPoint();
};

#endif