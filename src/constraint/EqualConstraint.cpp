#include "EqualConstraint.hpp"

EqualConstraint::EqualConstraint(const Vertex* vertex, const Vector3f& position) :
    vertex(const_cast<Vertex*>(vertex)),
    position(position) {}

EqualConstraint::~EqualConstraint() {}

std::vector<Gradient*> EqualConstraint::energyGradient() const {
    std::vector<Gradient*> ans;
    ans.push_back(new Gradient(vertex->index, 1000.0f * (vertex->x - position)));
    return ans;
}

std::vector<Hessian*> EqualConstraint::energyHessian() const {
    std::vector<Hessian*> ans;
    ans.push_back(new Hessian(vertex->index, vertex->index, 1000.0f * Matrix3x3f::Identity()));
    return ans;
}
