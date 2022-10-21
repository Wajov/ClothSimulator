#ifndef NEAR_POINT_HPP
#define NEAR_POINT_HPP

#include "TypeHelper.hpp"

class NearPoint {
public:
    float d;
    Vector3f x;
    NearPoint(float d, const Vector3f& x);
    ~NearPoint();
};

#endif