#ifndef CONSTRAINT_CUH
#define CONSTRAINT_CUH

#include <vector>

#include "Gradient.cuh"
#include "Hessian.cuh"

class Constraint {
public:
    Constraint();
    ~Constraint();
    virtual std::vector<Gradient*> energyGradient() const = 0;
    virtual std::vector<Hessian*> energyHessian() const = 0;
};

#endif