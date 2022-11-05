#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include "Vector.hpp"

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