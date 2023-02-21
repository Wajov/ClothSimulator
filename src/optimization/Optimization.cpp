#include "Optimization.hpp"

Optimization::Optimization() {}

Optimization::~Optimization() {}

int Optimization::getNodeSize() const {
    return nodeSize;
}

int Optimization::getConstraintSize() const {
    return constraintSize;
}
