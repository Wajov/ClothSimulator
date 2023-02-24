#include "Plane.cuh"

Plane::Plane(const Vector3f& p, const Vector3f& n) :
    p(p),
    n(n) {}

Plane::~Plane() {}