#ifndef HESSIAN_HPP
#define HESSIAN_HPP

#include "Vector.hpp"
#include "Matrix.hpp"

class Hessian {
private:
    int index0, index1;
    Matrix3x3f value;

public:
    Hessian(int index0, int index1, const Matrix3x3f& value);
    ~Hessian();
    int getIndex0() const;
    int getIndex1() const;
    Matrix3x3f getValue() const;
};

#endif