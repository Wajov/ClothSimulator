#include "Intersection.hpp"

Intersection::Intersection(const Face* face0, const Face* face1, const Vector3f& b0, const Vector3f& b1, const Vector3f& d) :
    face0(const_cast<Face*>(face0)),
    face1(const_cast<Face*>(face1)),
    b0(b0),
    b1(b1),
    d(d) {}

Intersection::~Intersection() {}