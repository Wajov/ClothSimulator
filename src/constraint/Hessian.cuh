#ifndef HESSIAN_CUH
#define HESSIAN_CUH

#include "Vector.cuh"
#include "Matrix.cuh"

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