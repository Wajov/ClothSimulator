#include "EqualConstraint.cuh"

EqualConstraint::EqualConstraint(const Node* node, const Vector3f& position) :
    node(const_cast<Node*>(node)),
    position(position) {}

EqualConstraint::~EqualConstraint() {}

std::vector<Gradient*> EqualConstraint::energyGradient() const {
    std::vector<Gradient*> ans;
    ans.push_back(new Gradient(node->index, 1000.0f * (node->x - position)));
    return ans;
}

std::vector<Hessian*> EqualConstraint::energyHessian() const {
    std::vector<Hessian*> ans;
    ans.push_back(new Hessian(node->index, node->index, 1000.0f * Matrix3x3f(1.0f)));
    return ans;
}
