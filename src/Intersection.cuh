#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "Vector.cuh"
#include "Face.cuh"

class Intersection {
public:
    Face* face0, * face1;
    Vector3f b0, b1, d;
    Intersection(const Face* face0, const Face* face1, const Vector3f& b0, const Vector3f& b1, const Vector3f& d);
    ~Intersection();
};

#endif