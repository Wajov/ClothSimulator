#include "Optimization.hpp"

Optimization::Optimization() {}

Optimization::~Optimization() {}

int Optimization::getVariableSize() {
    return variableSize;
}

int Optimization::getConstraintSize() {
    return constraintSize;
}
