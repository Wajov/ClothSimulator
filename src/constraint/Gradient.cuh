#ifndef GRADIENT_CUH
#define GRADIENT_CUH

#include "Vector.cuh"

class Gradient {
private:
    int index;
    Vector3f value;

public:
    Gradient(int index, const Vector3f& value);
    ~Gradient();
    int getIndex() const;
    Vector3f getValue() const;
};

#endif