#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include <vector>

#include "Gradient.hpp"
#include "Hessian.hpp"

class Constraint {
public:
    Constraint();
    ~Constraint();
    virtual std::vector<Gradient*> energyGradient() const = 0;
    virtual std::vector<Hessian*> energyHessian() const = 0;
};

#endif