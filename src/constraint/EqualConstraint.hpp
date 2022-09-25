#ifndef EQUAL_CONSTRAINT_HPP
#define EQUAL_CONSTRAINT_HPP

#include "TypeHelper.hpp"
#include "Vertex.hpp"
#include "Constraint.hpp"

class EqualConstraint : public Constraint {
private:
    Vertex* vertex;
    Vector3f position;

public:
    EqualConstraint(const Vertex* vertex, const Vector3f& position);
    ~EqualConstraint();
    std::vector<Gradient*> energyGradient() const override;
    std::vector<Hessian*> energyHessian() const override;
};

#endif