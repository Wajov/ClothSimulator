#include "Hessian.hpp"

Hessian::Hessian(int index0, int index1, const Matrix3x3f& value) :
    index0(index0),
    index1(index1),
    value(value) {}

Hessian::~Hessian() {}

int Hessian::getIndex0() const {
    return index0;    
}

int Hessian::getIndex1() const {
    return index1;    
}

Matrix3x3f Hessian::getValue() const {
    return value;    
}
