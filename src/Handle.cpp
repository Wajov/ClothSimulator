#include "Handle.hpp"

Handle::Handle(const Vertex* vertex, const Vector3f& position) :
    vertex(const_cast<Vertex*>(vertex)),
    position(position) {}

Handle::~Handle() {}

Constraint* Handle::getConstraint() const {
    return new EqualConstraint(vertex, position);
}

Vertex* Handle::getVertex() const {
    return vertex;
}

Vector3f Handle::getPosition() const {
    return position;
}
