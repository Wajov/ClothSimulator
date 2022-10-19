#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include <optimization.h>

class Optimization {
protected:
    int variableSize, constraintSize;

public:
    Optimization();
    virtual ~Optimization();
    int getVariableSize();
    int getConstraintSize();
    virtual void initialize(double* x) const = 0;
    virtual void precompute(const double *x) = 0;
    virtual void finalize(const double* x) = 0;
    virtual double objective(const double* x) const = 0;
    virtual void objectiveGradient(const double* x, double* gradient) const = 0;
    virtual double constraint(const double* x, int index, int& sign) const = 0;
    virtual void constraintGradient(const double* x, int index, double factor, double* gradient) const = 0;
};

#endif