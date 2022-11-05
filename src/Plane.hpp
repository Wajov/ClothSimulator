#ifndef PLANE_HPP
#define PLANE_HPP

#include "Vector.hpp"

class Plane {
public:
    Vector3f p, n;
    Plane(const Vector3f& p, const Vector3f& n);
    ~Plane();
};

#endif