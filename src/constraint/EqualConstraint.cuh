#ifndef EQUAL_CONSTRAINT_CUH
#define EQUAL_CONSTRAINT_CUH

#include "Vector.cuh"
#include "Node.cuh"
#include "Constraint.cuh"

class EqualConstraint : public Constraint {
private:
    Node* node;
    Vector3f position;

public:
    EqualConstraint(const Node* node, const Vector3f& position);
    ~EqualConstraint();
    std::vector<Gradient*> energyGradient() const override;
    std::vector<Hessian*> energyHessian() const override;
};

#endif