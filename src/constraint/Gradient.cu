#include "Gradient.cuh"

Gradient::Gradient(int index, const Vector3f& value) :
    index(index),
    value(value) {}

Gradient::~Gradient() {}

int Gradient::getIndex() const {
    return index;    
}

Vector3f Gradient::getValue() const {
    return value;    
}
